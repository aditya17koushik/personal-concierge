"""The coach's judgement, in code.

Every verdict ("you are under-eating", "6 days without rest") is a named flag computed here from
the user's logs. The model never decides whether someone is over-eating; it only words the flags
it is given. Flags are conservative: they need enough data, and a safety rule stops the coach
from pushing a person who may already be eating or training too little or too much.
"""

from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Literal

Level = Literal["good", "info", "warn", "alert"]
Scope = Literal["trend", "today"]

MIN_LOGGED_DAYS = 3  # fewer logged days than this and we do not judge intake
PLAN_REVIEW_DAYS = 14  # suggest a plan update after this long
VERY_LOW_AVG_KCAL = 1200  # a logged average below this triggers the safety rule
UNDER_EATING = {"cut": 0.75, "maintain": 0.75, "bulk": 0.90}  # avg / target below this
OVER_EATING = {"cut": 1.10, "maintain": 1.10, "bulk": 1.20}  # avg / target above this
LOW_PROTEIN_RATIO = 0.80
STREAK_WARN = 6
STREAK_ALERT = 7  # a whole week without a rest day
HIGH_VOLUME_MINUTES = 600  # in 7 days
STALE_WEIGH_IN_DAYS = 10
PROTEIN_IDEAS = "eggs, curd or Greek yogurt, paneer or tofu, chicken or fish, dal, or a whey shake"

_ORDER = {"alert": 0, "warn": 1, "info": 2, "good": 3}


@dataclass(frozen=True)
class DayStat:
    day: date
    calories: int = 0
    meals: int = 0
    protein_g: float = 0.0
    protein_known: bool = True  # every meal that day has a protein number
    workouts: int = 0
    minutes: int = 0


@dataclass(frozen=True)
class CoachData:
    today: date
    days: list[DayStat]  # the last 7 days, oldest first, ending today
    weights: list[tuple[date, float]]  # (day, kg) over the last ~28 days, oldest first
    mode: str | None = None  # cut | bulk | maintain, or None when no goal is set
    target_calories: int | None = None
    target_protein_g: int | None = None
    weekly_workouts: int | None = None
    goal_started_on: date | None = None
    local_hour: int | None = None  # set for the evening check-in
    unit: str = "kg"
    plan_age_days: int | None = None  # age of the active plan, None when there is no plan


@dataclass(frozen=True)
class Flag:
    code: str
    level: Level
    text: str  # what is going on, with numbers
    suggest: str | None = None  # one concrete next step
    scope: Scope = "trend"
    priority: int = 50  # lower first within a level

    def line(self) -> str:
        return self.text + (f" {self.suggest}" if self.suggest else "")


@dataclass
class Review:
    flags: list[Flag] = field(default_factory=list)
    facts: dict[str, object] = field(default_factory=dict)

    def by_level(self, *levels: Level) -> list[Flag]:
        return [f for f in self.flags if f.level in levels]


def _kg(value: float, unit: str) -> str:
    shown = value / 0.45359237 if unit == "lb" else value
    return f"{abs(shown):.1f} {unit}"


def consecutive_training_days(days: list[DayStat]) -> int:
    """Run of trained days ending today, or yesterday if today has no workout yet."""
    ordered = list(reversed(days))  # newest first
    if ordered and ordered[0].workouts == 0:
        ordered = ordered[1:]
    count = 0
    for stat in ordered:
        if stat.workouts == 0:
            break
        count += 1
    return count


def _weight_trend(data: CoachData) -> tuple[float, float] | None:
    """(change in kg per week, latest kg) when there are two weigh-ins a week or more apart."""
    if len(data.weights) < 2:
        return None
    (d0, w0), (d1, w1) = data.weights[0], data.weights[-1]
    span = (d1 - d0).days
    if span < 7:
        return None
    return (w1 - w0) / (span / 7), w1


def evaluate(data: CoachData, scope: Scope = "trend") -> Review:
    """Judge the data. scope='today' adds the same-day checks used by the evening check-in."""
    review = Review()
    flags = review.flags
    mode = data.mode
    past = [d for d in data.days if d.day < data.today]
    logged = [d for d in past if d.meals > 0]
    window = data.days
    sessions = sum(d.workouts for d in window)
    minutes = sum(d.minutes for d in window)
    trained_days = sum(1 for d in window if d.workouts)

    avg_cal: float | None = None
    if len(logged) >= MIN_LOGGED_DAYS:
        avg_cal = sum(d.calories for d in logged) / len(logged)

    review.facts.update(
        logged_days=len(logged),
        past_days=len(past),
        avg_calories=round(avg_cal) if avg_cal is not None else None,
        sessions_7d=sessions,
        training_days_7d=trained_days,
        training_minutes_7d=minutes,
        mode=mode,
    )

    # --- safety first: a very low logged intake stops all pushing
    if avg_cal is not None and avg_cal < VERY_LOW_AVG_KCAL:
        flags.append(
            Flag(
                "very_low_intake",
                "alert",
                f"Your logged intake averages {round(avg_cal)} kcal a day over {len(logged)} days, "
                "which is very low.",
                "If that is accurate, please talk to a doctor or dietitian before changing "
                "anything else. If you are just not logging everything, log every meal so I "
                "can coach you properly.",
                priority=0,
            )
        )
        if consecutive_training_days(window) >= STREAK_WARN or trained_days == 7:
            flags.append(_streak_flag(consecutive_training_days(window), recovery=True))
        review.flags = _sorted(flags)
        return review  # nothing that pushes the person to eat less or train more

    # --- intake against the calorie target
    if avg_cal is not None and data.target_calories and mode:
        ratio = avg_cal / data.target_calories
        under, over = UNDER_EATING[mode], OVER_EATING[mode]
        gap = round(data.target_calories - avg_cal)
        pct = round(ratio * 100)
        if ratio < under:
            why = (
                "Too little food slows recovery and muscle gain."
                if mode == "bulk"
                else "Eating well under target hurts training and recovery, and a deficit that is "
                "too big is hard to keep up."
            )
            flags.append(
                Flag(
                    "under_eating",
                    "warn",
                    f"Under-eating: you average {round(avg_cal)} kcal on logged days, {pct}% of "
                    f"your {data.target_calories} kcal target.",
                    f"{why} Add about {min(gap, 500)} kcal a day, ideally with protein.",
                    priority=10,
                )
            )
        elif ratio > over:
            excess = round(avg_cal - data.target_calories)
            effect = {
                "cut": "At this level the cut will stall.",
                "bulk": "Much of the extra will go on as fat.",
                "maintain": "Your weight will drift up.",
            }[mode]
            flags.append(
                Flag(
                    "over_eating",
                    "warn",
                    f"Over-eating: you average {round(avg_cal)} kcal on logged days, {pct}% of "
                    f"your {data.target_calories} kcal target (about {excess} over).",
                    f"{effect} Trim about {min(excess, 400)} kcal a day: smaller portions of "
                    "the calorie-dense foods, or fewer snacks.",
                    priority=10,
                )
            )
        else:
            flags.append(
                Flag(
                    "intake_on_target",
                    "good",
                    f"Intake is on target: {round(avg_cal)} kcal on logged days against "
                    f"{data.target_calories}.",
                    priority=60,
                )
            )

    # --- protein (only judged when every logged meal has a protein number)
    if data.target_protein_g and len(logged) >= MIN_LOGGED_DAYS and all(d.protein_known for d in logged):
        avg_protein = sum(d.protein_g for d in logged) / len(logged)
        if avg_protein < data.target_protein_g * LOW_PROTEIN_RATIO:
            short = round(data.target_protein_g - avg_protein)
            flags.append(
                Flag(
                    "low_protein",
                    "warn",
                    f"Protein is low: about {round(avg_protein)} g a day against your "
                    f"{data.target_protein_g} g target.",
                    f"You are about {short} g short. Easy fixes: {PROTEIN_IDEAS}.",
                    priority=20,
                )
            )

    if len(past) >= 4 and len(logged) < len(past) - 2 and len(logged) > 0:
        flags.append(
            Flag(
                "sparse_logging",
                "info",
                f"Meals were logged on only {len(logged)} of the last {len(past)} days, so "
                "the food numbers are rough.",
                "Logging everything, even roughly, makes the advice much better.",
                priority=40,
            )
        )

    # --- training load
    streak = consecutive_training_days(window)
    heavy = trained_days == 7 or minutes >= HIGH_VOLUME_MINUTES or (
        data.weekly_workouts is not None and sessions >= max(8, data.weekly_workouts + 4)
    )
    if streak >= STREAK_WARN:
        flags.append(_streak_flag(streak, recovery=False))
    elif heavy:
        flags.append(
            Flag(
                "high_training_load",
                "warn",
                f"Heavy week: {sessions} sessions and {minutes} minutes in the last 7 days.",
                "Plan at least one full rest day or a light walk or stretch day.",
                priority=15,
            )
        )
    if (streak >= STREAK_WARN or heavy) and any(f.code in ("under_eating",) for f in flags):
        flags.append(
            Flag(
                "recovery_risk",
                "alert",
                "High training load with low intake is a recovery risk: soreness, poor sleep "
                "and stalled progress follow.",
                "Eat up to your target and take a rest day before the next hard session.",
                priority=1,
            )
        )

    goal_age = (data.today - data.goal_started_on).days if data.goal_started_on else 0
    if data.weekly_workouts is not None and goal_age >= 6:
        if sessions + 1 < data.weekly_workouts:
            flags.append(
                Flag(
                    "under_training",
                    "warn",
                    f"Under-training: {sessions} session{'s' if sessions != 1 else ''} in the "
                    f"last 7 days against a target of {data.weekly_workouts}.",
                    "Book the next one for today or tomorrow. Even a 20 minute walk keeps the "
                    "habit alive.",
                    priority=25,
                )
            )
        elif not heavy and streak < STREAK_WARN:
            flags.append(
                Flag(
                    "training_on_track",
                    "good",
                    f"Training is on track: {sessions} sessions in the last 7 days "
                    f"(target {data.weekly_workouts}).",
                    priority=60,
                )
            )

    # --- weight trend
    trend = _weight_trend(data)
    if data.weights:
        review.facts["weight_latest_kg"] = data.weights[-1][1]
    if trend is not None:
        review.facts["weight_per_week_kg"] = round(trend[0], 2)
    if trend is not None and mode:
        per_week, latest = trend
        pct_week = per_week / latest * 100
        rate = _kg(per_week, data.unit) + "/week"
        intake_high = any(f.code == "over_eating" for f in flags)
        if pct_week <= -1.0:
            flags.append(
                Flag(
                    "losing_too_fast",
                    "warn",
                    f"Weight is dropping {rate}, faster than about 1% of body weight a week.",
                    "Slow down: eat a little more so you keep muscle and energy.",
                    priority=12,
                )
            )
        elif mode == "cut" and pct_week >= 0.2:
            flags.append(
                Flag(
                    "weight_wrong_way",
                    "warn",
                    f"Weight is going up ({rate}) while your goal is a cut.",
                    "Check that you log everything, then bring intake down"
                    if not intake_high
                    else "Your intake is above target, which explains it. Bring it down.",
                    priority=11,
                )
            )
        elif mode == "cut" and pct_week > -0.2:
            flags.append(
                Flag(
                    "cut_stalled",
                    "warn",
                    "Weight has not moved over the last weeks, so the cut has stalled.",
                    "Your intake is above target, which explains it. Bring it down."
                    if intake_high
                    else "First check your logging is complete. If it is accurate for two "
                    "weeks, lower calories by about 100-150 a day or add daily walking.",
                    priority=11,
                )
            )
        elif mode == "bulk" and pct_week <= -0.1:
            flags.append(
                Flag(
                    "weight_wrong_way",
                    "warn",
                    f"Weight is going down ({rate}) while your goal is a bulk.",
                    "Add about 200-300 kcal a day.",
                    priority=11,
                )
            )
        elif mode == "bulk" and 0.1 > pct_week > -0.1:
            flags.append(
                Flag(
                    "bulk_stalled",
                    "warn",
                    "Weight has not moved over the last weeks, so the bulk has stalled.",
                    "If your logging is complete, add about 150-200 kcal a day.",
                    priority=11,
                )
            )
        elif mode == "bulk" and pct_week >= 0.75:
            flags.append(
                Flag(
                    "bulking_too_fast",
                    "info",
                    f"Weight is rising {rate}, quicker than about 0.5% a week, so more of it "
                    "may be fat.",
                    "Trim the surplus slightly.",
                    priority=30,
                )
            )
        elif mode == "maintain" and abs(pct_week) >= 0.5:
            direction = "up" if pct_week > 0 else "down"
            flags.append(
                Flag(
                    "weight_drifting",
                    "info",
                    f"Weight is drifting {direction} ({rate}) while you are maintaining.",
                    "Nudge calories back toward your target.",
                    priority=30,
                )
            )
        else:
            flags.append(
                Flag(
                    "weight_on_track",
                    "good",
                    f"Weight trend is on track for a {mode}: {rate}."
                    if mode != "maintain"
                    else "Weight is stable, as planned.",
                    priority=60,
                )
            )

    if mode in ("cut", "bulk") and data.weights:
        days_since = (data.today - data.weights[-1][0]).days
        if days_since >= STALE_WEIGH_IN_DAYS:
            flags.append(
                Flag(
                    "weigh_in_stale",
                    "info",
                    f"Your last weigh-in was {days_since} days ago.",
                    "Weigh in this week so I can track the trend.",
                    priority=45,
                )
            )

    if mode is None:
        flags.append(
            Flag(
                "no_goal",
                "info",
                "You have not set a goal yet.",
                "Tell me: I want to cut, bulk or maintain.",
                priority=5,
            )
        )
    elif data.target_calories is None:
        flags.append(
            Flag(
                "no_calorie_target",
                "info",
                f"You are on a {mode} but have no daily calorie target, so I cannot judge "
                "your eating.",
                "Tell me a daily calorie target, for example: set my calories to 2000.",
                priority=6,
            )
        )

    if scope == "today":
        flags.extend(_today_flags(data))

    if data.plan_age_days is not None and data.plan_age_days >= PLAN_REVIEW_DAYS:
        adapt = {
            "cut_stalled", "weight_wrong_way", "bulk_stalled", "losing_too_fast",
            "bulking_too_fast", "weight_drifting", "high_training_load", "recovery_risk",
            "under_training",
        }  # fmt: skip
        if any(f.code in adapt for f in flags):
            flags.append(
                Flag(
                    "plan_update_suggested",
                    "info",
                    f"Your plan is {data.plan_age_days} days old and your progress suggests it "
                    "needs adjusting.",
                    "Say 'update my plan' and I will adjust training and meals.",
                    priority=20,
                )
            )

    review.flags = _sorted(flags)
    return review


def _streak_flag(streak: int, recovery: bool) -> Flag:
    level: Level = "alert" if streak >= STREAK_ALERT else "warn"
    extra = " With intake this low, rest matters even more." if recovery else ""
    return Flag(
        "training_streak",
        level,
        f"You have trained {streak} days in a row.{extra}",
        "Take a full rest day. Muscle grows and performance returns while you recover.",
        priority=14,
    )


def _today_flags(data: CoachData) -> list[Flag]:
    out: list[Flag] = []
    today = next((d for d in data.days if d.day == data.today), DayStat(data.today))
    hour = data.local_hour if data.local_hour is not None else 12
    target = data.target_calories

    if today.meals == 0 and hour >= 14:
        out.append(
            Flag(
                "nothing_logged_today",
                "info",
                "No meals are logged today.",
                "Log what you have eaten, even roughly, so today counts.",
                scope="today",
                priority=3,
            )
        )
    elif target and hour >= 19 and today.meals > 0 and today.calories < target * 0.6:
        short = target - today.calories
        out.append(
            Flag(
                "calories_short_today",
                "warn",
                f"You are at {today.calories} kcal of {target} today, {short} short.",
                "Have a proper meal with protein tonight rather than going to bed under-fuelled.",
                scope="today",
                priority=8,
            )
        )
    elif target and today.calories > target * 1.15:
        over = today.calories - target
        out.append(
            Flag(
                "calories_over_today",
                "info",
                f"You are {over} kcal over target today.",
                "One day does not change your trend. Keep tomorrow on target.",
                scope="today",
                priority=9,
            )
        )

    if (
        data.target_protein_g
        and hour >= 19
        and today.meals > 0
        and today.protein_known
        and today.protein_g < data.target_protein_g - 30
    ):
        gap = round(data.target_protein_g - today.protein_g)
        out.append(
            Flag(
                "protein_gap_today",
                "info",
                f"Protein is at {round(today.protein_g)} g of {data.target_protein_g} g today.",
                f"About {gap} g to go: {PROTEIN_IDEAS}.",
                scope="today",
                priority=18,
            )
        )
    return out


def _sorted(flags: list[Flag]) -> list[Flag]:
    return sorted(flags, key=lambda f: (_ORDER[f.level], f.priority))


def rest_days_left(days: list[DayStat], today: date) -> int:
    """How many of the last 7 days had no workout (used in the weekly summary)."""
    return sum(1 for d in days if d.day <= today and d.workouts == 0)


def last_seven(today: date) -> list[date]:
    return [today - timedelta(days=n) for n in range(6, -1, -1)]