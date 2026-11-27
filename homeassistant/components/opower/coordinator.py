"""Coordinator to handle Opower connections."""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
from typing import Any, cast, override

from opower import (
    Account,
    AggregateType,
    CostRead,
    Forecast,
    MeterType,
    Opower,
    ReadResolution,
    create_cookie_jar,
)
from opower.exceptions import ApiException, CannotConnect, InvalidAuth, MfaChallenge

from homeassistant.components.recorder import get_instance
from homeassistant.components.recorder.models import (
    StatisticData,
    StatisticMeanType,
    StatisticMetaData,
)
from homeassistant.components.recorder.statistics import (
    async_add_external_statistics,
    get_last_statistics,
    statistics_during_period,
)
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import CONF_PASSWORD, CONF_USERNAME, UnitOfEnergy, UnitOfVolume
from homeassistant.core import HomeAssistant, callback
from homeassistant.exceptions import ConfigEntryAuthFailed
from homeassistant.helpers import issue_registry as ir
from homeassistant.helpers.aiohttp_client import async_create_clientsession
from homeassistant.helpers.update_coordinator import DataUpdateCoordinator, UpdateFailed
from homeassistant.util import dt as dt_util
from homeassistant.util.unit_conversion import EnergyConverter, VolumeConverter

from .const import CONF_LOGIN_DATA, CONF_TOTP_SECRET, CONF_UTILITY, DOMAIN

_LOGGER = logging.getLogger(__name__)

# Lookback for late-priced cost, on top of the 30 days re-read every update (the
# library caps hourly reads at 60 days: up to 60 days for HOUR, 90 for DAY).
# Longer lags are not reached for: such cost is only published per bill.
MAX_COST_LAG = timedelta(days=60)

type OpowerConfigEntry = ConfigEntry[OpowerCoordinator]


@dataclass
class _StatisticPair:
    """Rows for a pair of statistics and the last start of each, if stored."""

    rows: list[tuple[datetime, float, float]]
    last_start: float | None
    opposite_last_start: float | None
    metadata: StatisticMetaData
    opposite_metadata: StatisticMetaData
    # Starts of reads in the window without a row here (unpriced reads)
    gaps: list[datetime] = field(default_factory=list)

    @property
    def last_stat_start(self) -> float | None:
        """Return the start of the last statistic of the pair, if any."""
        return max(
            (
                start
                for start in (self.last_start, self.opposite_last_start)
                if start is not None
            ),
            default=None,
        )


@dataclass
class _Baseline:
    """Where a pair of statistics resumes from.

    Rows at or before `after` (`opposite_after`) are already stored for the
    statistic (its opposite); None means every row is written. `kept` holds
    stored rows to write back for reads that came without a value.
    """

    total: float
    opposite_total: float
    after: float | None
    opposite_after: float | None
    kept: list[tuple[datetime, float, float]] = field(default_factory=list)


def _should_update(pair: _StatisticPair) -> bool:
    """Whether a pair has rows that can be written without a discontinuity."""
    if not pair.rows:
        return False
    if (
        pair.last_stat_start is not None
        and pair.rows[-1][0].timestamp() < pair.last_stat_start
    ):
        # Rewriting up to rows[-1] would break the sums of the newer statistics
        _LOGGER.debug(
            "Reads for %s end before its last statistic; skipping",
            pair.metadata["statistic_id"],
        )
        return False
    return True


@dataclass
class OpowerData:
    """Class to hold Opower data."""

    account: Account
    forecast: Forecast | None
    last_changed: datetime | None
    last_updated: datetime


class OpowerCoordinator(DataUpdateCoordinator[dict[str, OpowerData]]):
    """Handle fetching Opower data, updating sensors and inserting statistics."""

    config_entry: OpowerConfigEntry

    def __init__(
        self,
        hass: HomeAssistant,
        config_entry: OpowerConfigEntry,
    ) -> None:
        """Initialize the data handler."""
        super().__init__(
            hass,
            _LOGGER,
            config_entry=config_entry,
            name="Opower",
            # Data is updated daily on Opower.
            # Refresh every 12h to be at most 12h behind.
            update_interval=timedelta(hours=12),
        )
        self.api = Opower(
            async_create_clientsession(hass, cookie_jar=create_cookie_jar()),
            config_entry.data[CONF_UTILITY],
            config_entry.data[CONF_USERNAME],
            config_entry.data[CONF_PASSWORD],
            config_entry.data.get(CONF_TOTP_SECRET),
            config_entry.data.get(CONF_LOGIN_DATA),
        )

        @callback
        def _dummy_listener() -> None:
            pass

        # Force the coordinator to periodically update by
        # registering at least one listener.
        # Needed when the _async_update_data below returns {}
        # for utilities that don't provide forecast, which results
        # to no sensors added, no registered listeners, and thus
        # _async_update_data not periodically getting called which
        # is needed for _insert_statistics.
        self.async_add_listener(_dummy_listener)

    @override
    async def _async_update_data(
        self,
    ) -> dict[str, OpowerData]:
        """Fetch data from API endpoint."""
        try:
            # Login expires after a few minutes.
            # Given the infrequent updating (every 12h)
            # assume previous session has expired and re-login.
            await self.api.async_login()
        except (InvalidAuth, MfaChallenge) as err:
            raise ConfigEntryAuthFailed from err
        except CannotConnect as err:
            raise UpdateFailed(
                translation_domain=DOMAIN,
                translation_key="login_error",
                translation_placeholders={"error": str(err)},
            ) from err

        try:
            accounts = await self.api.async_get_accounts()
        except ApiException as err:
            _LOGGER.error("Error getting accounts: %s", err)
            raise

        try:
            forecasts_list = await self.api.async_get_forecast()
        except ApiException as err:
            _LOGGER.error("Error getting forecasts: %s", err)
            raise

        forecasts = {f.account.utility_account_id: f for f in forecasts_list}
        _LOGGER.debug("Updating sensor data with: %s", forecasts)

        # Because Opower provides historical usage/cost with a delay of a couple of days
        # we need to insert data into statistics.
        last_changed_per_account = await self._insert_statistics(accounts)
        return {
            account.utility_account_id: OpowerData(
                account=account,
                forecast=forecasts.get(account.utility_account_id),
                last_changed=last_changed_per_account.get(account.utility_account_id),
                last_updated=dt_util.utcnow(),
            )
            for account in accounts
        }

    async def _insert_statistics(self, accounts: list[Account]) -> dict[str, datetime]:
        """Insert Opower statistics."""
        last_changed_per_account: dict[str, datetime] = {}
        for account in accounts:
            id_prefix = (
                (
                    f"{self.api.utility.subdomain()}_{account.meter_type.name}_"
                    f"{account.utility_account_id}"
                )
                # Some utilities like AEP have "-" in their account id.
                # Other utilities like ngny-gas have "-" in their subdomain.
                # Replace it with "_" to avoid "Invalid statistic_id"
                .replace("-", "_")
                .lower()
            )
            cost_statistic_id = f"{DOMAIN}:{id_prefix}_energy_cost"
            compensation_statistic_id = f"{DOMAIN}:{id_prefix}_energy_compensation"
            consumption_statistic_id = f"{DOMAIN}:{id_prefix}_energy_consumption"
            return_statistic_id = f"{DOMAIN}:{id_prefix}_energy_return"
            _LOGGER.debug(
                "Updating Statistics for %s, %s, %s, and %s",
                cost_statistic_id,
                compensation_statistic_id,
                consumption_statistic_id,
                return_statistic_id,
            )

            name_prefix = (
                f"Opower {self.api.utility.subdomain()} "
                f"{account.meter_type.name.lower()} {account.utility_account_id}"
            )
            cost_metadata = StatisticMetaData(
                mean_type=StatisticMeanType.NONE,
                has_sum=True,
                name=f"{name_prefix} cost",
                source=DOMAIN,
                statistic_id=cost_statistic_id,
                unit_class=None,
                unit_of_measurement=None,
            )
            compensation_metadata = StatisticMetaData(
                mean_type=StatisticMeanType.NONE,
                has_sum=True,
                name=f"{name_prefix} compensation",
                source=DOMAIN,
                statistic_id=compensation_statistic_id,
                unit_class=None,
                unit_of_measurement=None,
            )
            consumption_unit_class = (
                EnergyConverter.UNIT_CLASS
                if account.meter_type is MeterType.ELEC
                else VolumeConverter.UNIT_CLASS
            )
            consumption_unit = (
                UnitOfEnergy.KILO_WATT_HOUR
                if account.meter_type is MeterType.ELEC
                else UnitOfVolume.CENTUM_CUBIC_FEET
            )
            consumption_metadata = StatisticMetaData(
                mean_type=StatisticMeanType.NONE,
                has_sum=True,
                name=f"{name_prefix} consumption",
                source=DOMAIN,
                statistic_id=consumption_statistic_id,
                unit_class=consumption_unit_class,
                unit_of_measurement=consumption_unit,
            )
            return_metadata = StatisticMetaData(
                mean_type=StatisticMeanType.NONE,
                has_sum=True,
                name=f"{name_prefix} return",
                source=DOMAIN,
                statistic_id=return_statistic_id,
                unit_class=consumption_unit_class,
                unit_of_measurement=consumption_unit,
            )

            last_starts = await get_instance(self.hass).async_add_executor_job(
                self._get_last_stat_starts,
                [
                    consumption_statistic_id,
                    return_statistic_id,
                    cost_statistic_id,
                    compensation_statistic_id,
                ],
            )
            last_consumption_start = last_starts[consumption_statistic_id]
            last_cost_start = last_starts[cost_statistic_id]
            if last_consumption_start is not None:
                # Kept when this update writes nothing, so it never goes backward
                last_changed_per_account[account.utility_account_id] = (
                    dt_util.utc_from_timestamp(last_consumption_start)
                )
            if last_consumption_start is None:
                _LOGGER.debug("Updating statistic for the first time")
                start_time = None
            else:
                migrated = await self._async_maybe_migrate_statistics(
                    account.utility_account_id,
                    {
                        cost_statistic_id: compensation_statistic_id,
                        consumption_statistic_id: return_statistic_id,
                    },
                    {
                        cost_statistic_id: cost_metadata,
                        compensation_statistic_id: compensation_metadata,
                        consumption_statistic_id: consumption_metadata,
                        return_statistic_id: return_metadata,
                    },
                )
                if migrated:
                    # Skip update to avoid working on old data since
                    # the migration is done
                    # asynchronously. Update the statistics in the next refresh in 12h.
                    _LOGGER.debug(
                        "Statistics migration completed. Skipping update for now"
                    )
                    continue
                start_time = last_consumption_start
                if (
                    last_cost_start is not None
                    and start_time - MAX_COST_LAG.total_seconds()
                    <= last_cost_start
                    < start_time
                ):
                    # Cost lags usage (the utility hasn't priced the latest reads
                    # yet): reach back far enough to fill it in.
                    start_time = last_cost_start
            cost_reads = await self._async_get_cost_reads(
                account, self.api.utility.timezone(), start_time
            )
            if not cost_reads:
                _LOGGER.debug("No recent usage/cost data. Skipping update")
                continue

            last_changed_per_account[account.utility_account_id] = max(
                cost_reads[-1].start_time,
                last_changed_per_account.get(
                    account.utility_account_id, cost_reads[-1].start_time
                ),
            )

            # Usage for every read; cost only for the reads the utility has priced.
            # Each pair resumes from its own statistics, so cost fills in once it's
            # published.
            consumption_rows = []
            for cost_read in cost_reads:
                # Prefer the meter's own import and export registers when the
                # utility publishes them. Splitting the net consumption on its
                # sign undercounts both directions on a net-metered site,
                # because an interval that is net-export can still contain real
                # import (and vice versa).
                if cost_read.imported is not None and cost_read.exported is not None:
                    consumption_rows.append(
                        (cost_read.start_time, cost_read.imported, cost_read.exported)
                    )
                else:
                    consumption_rows.append(
                        (
                            cost_read.start_time,
                            max(0, cost_read.consumption),
                            max(0, -cost_read.consumption),
                        )
                    )
            cost_rows = [
                (
                    cost_read.start_time,
                    max(0, cost_read.provided_cost),
                    max(0, -cost_read.provided_cost),
                )
                for cost_read in cost_reads
                if cost_read.provided_cost is not None
            ]
            if last_consumption_start is None and (
                last_cost_start is not None
                or last_starts[compensation_statistic_id] is not None
            ):
                # A consumption-only backfill fetches the full history at coarse
                # resolution (bill, then day, then hour). Writing that as cost would
                # overwrite finer cost statistics already stored, so leave cost to
                # the next incremental update, which reads it at native resolution.
                _LOGGER.debug(
                    "Backfilling consumption only; leaving %s as is", cost_statistic_id
                )
                cost_rows = []
            pairs = [
                _StatisticPair(
                    consumption_rows,
                    last_consumption_start,
                    last_starts[return_statistic_id],
                    consumption_metadata,
                    return_metadata,
                ),
                _StatisticPair(
                    cost_rows,
                    last_cost_start,
                    last_starts[compensation_statistic_id],
                    cost_metadata,
                    compensation_metadata,
                    [
                        cost_read.start_time
                        for cost_read in cost_reads
                        if cost_read.provided_cost is None
                    ],
                ),
            ]
            await self._async_add_statistics(
                [pair for pair in pairs if _should_update(pair)]
            )

        return last_changed_per_account

    def _get_last_stat_starts(
        self, statistic_ids: list[str]
    ) -> dict[str, float | None]:
        """Return the start of the last statistic of each statistic_id.

        Runs in the recorder executor; all lookups share one job.
        """
        starts: dict[str, float | None] = {}
        for statistic_id in statistic_ids:
            last_stat = get_last_statistics(self.hass, 1, statistic_id, True, set())
            starts[statistic_id] = (
                last_stat[statistic_id][0]["start"] if last_stat else None
            )
        return starts

    def _get_statistics(
        self, start: datetime, statistic_ids: set[str], required_ids: set[str]
    ) -> dict[str, list[Any]]:
        """Get the statistics at start, or else all of them from start on.

        Falls back to the open-ended query only if one of required_ids has no
        statistic at start.
        """
        stats: dict[str, list[Any]] = {}
        for end in (start + timedelta(seconds=1), None):
            stats = statistics_during_period(
                self.hass, start, end, statistic_ids, "hour", None, {"sum"}
            )
            if all(stats.get(statistic_id) for statistic_id in required_ids):
                break
            if end:
                _LOGGER.debug(
                    "Not found. Trying to find the oldest statistic after %s",
                    start,
                )
        return stats

    def _get_baseline(
        self, pair: _StatisticPair, stats: dict[str, list[Any]]
    ) -> _Baseline | None:
        """Return where a pair resumes from, given its statistics from its start.

        A statistic that had no rows when the update started is written from 0.
        None if one that had rows lost them since (cleared during the update):
        writing it from 0 for this window alone would leave its older history
        missing for good, so the pair waits for the next update.
        """
        statistic_id = pair.metadata["statistic_id"]
        opposite_statistic_id = pair.opposite_metadata["statistic_id"]
        primary = stats.get(statistic_id) or []
        opposite = stats.get(opposite_statistic_id) or []
        if (pair.last_start is not None and not primary) or (
            pair.last_start is None
            and pair.opposite_last_start is not None
            and not opposite
        ):
            return None
        anchor: float = (primary or opposite)[0]["start"]
        totals = []
        for records in (primary, opposite):
            total = 0.0
            for record in records:
                if record["start"] > anchor:
                    break
                total = float(record.get("sum") or 0.0)
            totals.append(total)
        return _Baseline(
            totals[0],
            totals[1],
            anchor if pair.last_start is not None else None,
            anchor if pair.opposite_last_start is not None else None,
        )

    def _get_kept_rows(
        self, pair: _StatisticPair, after: float
    ) -> list[tuple[datetime, float, float]]:
        """Return stored rows for reads in the window that came without a value.

        Without them the rows after such a read would be rewritten around a
        stored row that isn't, breaking the sums that follow.
        """
        gaps = [gap for gap in pair.gaps if gap.timestamp() > after]
        if not gaps or pair.last_stat_start is None:
            return []
        gaps = [gap for gap in gaps if gap.timestamp() <= pair.last_stat_start]
        if not gaps:
            return []
        statistic_id = pair.metadata["statistic_id"]
        opposite_statistic_id = pair.opposite_metadata["statistic_id"]
        stats = statistics_during_period(
            self.hass,
            gaps[0],
            gaps[-1] + timedelta(seconds=1),
            {statistic_id, opposite_statistic_id},
            "hour",
            None,
            {"state"},
        )
        opposite_states = {
            record["start"]: float(record.get("state") or 0.0)
            for record in stats.get(opposite_statistic_id, [])
        }
        gap_starts = {gap.timestamp(): gap for gap in gaps}
        return [
            (
                gap_starts[record["start"]],
                float(record.get("state") or 0.0),
                opposite_states.get(record["start"], 0.0),
            )
            for record in stats.get(statistic_id, [])
            if record["start"] in gap_starts
        ]

    def _get_baselines(self, pairs: list[_StatisticPair]) -> list[_Baseline | None]:
        """Return where each pair resumes from.

        Runs in the recorder executor, so all pairs share one job, and pairs that
        resume from the same time share one query.
        """
        # start -> (statistics to query, statistics the pairs resume from)
        by_start: dict[datetime, tuple[set[str], set[str]]] = {}
        starts: list[datetime | None] = []
        for pair in pairs:
            last_stat_start = pair.last_stat_start
            if last_stat_start is None:
                starts.append(None)
                continue
            start = min(pair.rows[0][0], dt_util.utc_from_timestamp(last_stat_start))
            starts.append(start)
            statistic_ids, required_ids = by_start.setdefault(start, (set(), set()))
            statistic_ids.update(
                (pair.metadata["statistic_id"], pair.opposite_metadata["statistic_id"])
            )
            # The pair resumes from the primary statistic unless it was cleared
            required_ids.add(
                pair.metadata["statistic_id"]
                if pair.last_start is not None
                else pair.opposite_metadata["statistic_id"]
            )
        stats_by_start: dict[datetime, dict[str, list[Any]]] = {}
        for start, (statistic_ids, required_ids) in by_start.items():
            _LOGGER.debug("Getting statistics at: %s", start)
            stats_by_start[start] = self._get_statistics(
                start, statistic_ids, required_ids
            )
        baselines: list[_Baseline | None] = []
        for pair, pair_start in zip(pairs, starts, strict=True):
            if pair_start is None:
                baselines.append(_Baseline(0.0, 0.0, None, None))
                continue
            baseline = self._get_baseline(pair, stats_by_start[pair_start])
            if baseline is not None and baseline.after is not None:
                baseline.kept = self._get_kept_rows(pair, baseline.after)
            baselines.append(baseline)
        return baselines

    async def _async_add_statistics(self, pairs: list[_StatisticPair]) -> None:
        """Add pairs of statistics (consumption/return, cost/compensation).

        Rows at or before the statistics they resume from are skipped; the rest
        replace what's there, so data the utility corrects or prices later is
        picked up.
        """
        if not pairs:
            return
        baselines = await get_instance(self.hass).async_add_executor_job(
            self._get_baselines, pairs
        )
        for pair, baseline in zip(pairs, baselines, strict=True):
            if baseline is None:
                _LOGGER.debug(
                    "Statistics for %s were removed during the update; skipping",
                    pair.metadata["statistic_id"],
                )
                continue
            total = baseline.total
            opposite_total = baseline.opposite_total
            statistics = []
            opposite_statistics = []
            for start, state, opposite_state in sorted(pair.rows + baseline.kept):
                timestamp = start.timestamp()
                if baseline.after is None or timestamp > baseline.after:
                    total += state
                    statistics.append(
                        StatisticData(start=start, state=state, sum=total)
                    )
                if (
                    baseline.opposite_after is None
                    or timestamp > baseline.opposite_after
                ):
                    opposite_total += opposite_state
                    opposite_statistics.append(
                        StatisticData(
                            start=start, state=opposite_state, sum=opposite_total
                        )
                    )
            for metadata, data in (
                (pair.metadata, statistics),
                (pair.opposite_metadata, opposite_statistics),
            ):
                if not data:
                    continue
                _LOGGER.debug(
                    "Adding %s statistics for %s", len(data), metadata["statistic_id"]
                )
                async_add_external_statistics(self.hass, metadata, data)

    async def _async_maybe_migrate_statistics(
        self,
        utility_account_id: str,
        migration_map: dict[str, str],
        metadata_map: dict[str, StatisticMetaData],
    ) -> bool:
        """Perform one-time statistics migration based on the provided map.

        Splits negative values from source IDs into target IDs.

        Args:
            utility_account_id: The account ID (for issue_id).
            migration_map: Map from source statistic ID to target statistic ID
                           (e.g., {cost_id: compensation_id}).
            metadata_map: Map of all statistic IDs (source and target)
                         to their metadata.

        """
        if not migration_map:
            return False

        need_migration_source_ids = await get_instance(
            self.hass
        ).async_add_executor_job(self._get_need_migration_source_ids, migration_map)
        if not need_migration_source_ids:
            return False

        _LOGGER.info("Starting one-time migration for: %s", need_migration_source_ids)

        processed_stats: dict[str, list[StatisticData]] = {}

        existing_stats = await get_instance(self.hass).async_add_executor_job(
            statistics_during_period,
            self.hass,
            dt_util.utc_from_timestamp(0),
            None,
            need_migration_source_ids,
            "hour",
            None,
            {"start", "state", "sum"},
        )
        for source_id, source_stats in existing_stats.items():
            _LOGGER.debug("Found %d statistics for %s", len(source_stats), source_id)
            if not source_stats:
                need_migration_source_ids.remove(source_id)
                continue
            target_id = migration_map[source_id]

            updated_source_stats: list[StatisticData] = []
            new_target_stats: list[StatisticData] = []
            updated_source_sum = 0.0
            new_target_sum = 0.0
            need_migration = False

            prev_sum = 0.0
            for stat in source_stats:
                start = dt_util.utc_from_timestamp(stat["start"])
                curr_sum = cast(float, stat["sum"])
                state = curr_sum - prev_sum
                prev_sum = curr_sum
                if state < 0:
                    need_migration = True

                updated_source_state = max(0, state)
                new_target_state = max(0, -state)

                updated_source_sum += updated_source_state
                new_target_sum += new_target_state

                updated_source_stats.append(
                    StatisticData(
                        start=start, state=updated_source_state, sum=updated_source_sum
                    )
                )
                new_target_stats.append(
                    StatisticData(
                        start=start, state=new_target_state, sum=new_target_sum
                    )
                )

            if need_migration:
                processed_stats[source_id] = updated_source_stats
                processed_stats[target_id] = new_target_stats
            else:
                need_migration_source_ids.remove(source_id)

        if not need_migration_source_ids:
            _LOGGER.debug("No migration needed")
            return False

        for stat_id, stats in processed_stats.items():
            _LOGGER.debug("Applying %d migrated stats for %s", len(stats), stat_id)
            async_add_external_statistics(self.hass, metadata_map[stat_id], stats)

        ir.async_create_issue(
            self.hass,
            DOMAIN,
            issue_id=f"return_to_grid_migration_{self.config_entry.entry_id}_{utility_account_id}",
            config_entry_id=self.config_entry.entry_id,
            is_fixable=False,
            severity=ir.IssueSeverity.WARNING,
            translation_key="return_to_grid_migration",
            translation_placeholders={
                "utility_account_id": utility_account_id,
                "energy_settings": "/config/energy",
                "target_ids": "\n".join(
                    {
                        str(metadata_map[v]["name"])
                        for k, v in migration_map.items()
                        if k in need_migration_source_ids
                    }
                ),
            },
        )

        return True

    def _get_need_migration_source_ids(self, migration_map: dict[str, str]) -> set[str]:
        """Return the source IDs that exist and whose target statistic does not yet."""
        need_migration_source_ids = set()
        for source_id, target_id in migration_map.items():
            # A source with no statistics (e.g. cost, when usage was imported before
            # the utility priced any of it) has nothing to migrate.
            if not get_last_statistics(self.hass, 1, source_id, True, set()):
                continue
            last_target_stat = get_last_statistics(
                self.hass,
                1,
                target_id,
                True,
                set(),
            )
            if not last_target_stat:
                need_migration_source_ids.add(source_id)
        return need_migration_source_ids

    async def _async_get_cost_reads(
        self, account: Account, time_zone_str: str, start_time: float | None = None
    ) -> list[CostRead]:
        """Get cost reads.

        If start_time is None, get cost reads since account activation,
        otherwise since start_time - 30 days to allow corrections in data from utilities

        We read at different resolutions depending on age:
        - month resolution for all years (since account activation)
        - day resolution for past 3 years (if account's read resolution supports it)
        - hour resolution for past 2 months (if account's read resolution supports it)
        """

        def _update_with_finer_cost_reads(
            cost_reads: list[CostRead], finer_cost_reads: list[CostRead]
        ) -> None:
            for i, cost_read in enumerate(cost_reads):
                for j, finer_cost_read in enumerate(finer_cost_reads):
                    if cost_read.start_time == finer_cost_read.start_time:
                        cost_reads[i:] = finer_cost_reads[j:]
                        return
                    if cost_read.end_time == finer_cost_read.start_time:
                        cost_reads[i + 1 :] = finer_cost_reads[j:]
                        return
                    if cost_read.end_time < finer_cost_read.start_time:
                        break
            cost_reads += finer_cost_reads

        tz = await dt_util.async_get_time_zone(time_zone_str)
        if start_time is None:
            start = None
        else:
            start = datetime.fromtimestamp(start_time, tz=tz) - timedelta(days=30)
        end = dt_util.now(tz)
        cost_reads: list[CostRead] = []
        if start_time is None or account.read_resolution is ReadResolution.BILLING:
            _LOGGER.debug("Getting monthly cost reads: %s - %s", start, end)
            try:
                cost_reads = await self.api.async_get_cost_reads(
                    account, AggregateType.BILL, start, end
                )
            except ApiException as err:
                _LOGGER.error("Error getting monthly cost reads: %s", err)
                raise
            _LOGGER.debug("Got %s monthly cost reads", len(cost_reads))
            if account.read_resolution is ReadResolution.BILLING:
                return cost_reads

        if start_time is None or account.read_resolution is ReadResolution.DAY:
            daily_start = end - timedelta(days=3 * 365)
            if start is not None:
                daily_start = max(start, daily_start)
            _LOGGER.debug("Getting daily cost reads: %s - %s", daily_start, end)
            try:
                daily_cost_reads = await self.api.async_get_cost_reads(
                    account, AggregateType.DAY, daily_start, end
                )
            except ApiException as err:
                _LOGGER.error("Error getting daily cost reads: %s", err)
                raise
            _LOGGER.debug("Got %s daily cost reads", len(daily_cost_reads))
            _update_with_finer_cost_reads(cost_reads, daily_cost_reads)
            if account.read_resolution is ReadResolution.DAY:
                return cost_reads

        hourly_start = end - timedelta(days=2 * 30)
        if start is not None:
            hourly_start = max(start, hourly_start)
        _LOGGER.debug("Getting hourly cost reads: %s - %s", hourly_start, end)
        try:
            hourly_cost_reads = await self.api.async_get_cost_reads(
                account, AggregateType.HOUR, hourly_start, end
            )
        except ApiException as err:
            _LOGGER.error("Error getting hourly cost reads: %s", err)
            raise
        _LOGGER.debug("Got %s hourly cost reads", len(hourly_cost_reads))
        _update_with_finer_cost_reads(cost_reads, hourly_cost_reads)
        _LOGGER.debug("Got %s cost reads", len(cost_reads))
        return cost_reads
