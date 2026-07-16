"""
Feature engineering module for datathon.
All feature transformations are safe to use inside CV folds
(i.e., they don't leak target information).
"""
import numpy as np
import pandas as pd
from scipy import stats


WEEK_COLS = [f'nilai_minggu_{i:02d}' for i in range(1, 13)]
ACTIVITY_COLS = [f'aktivitas_hari_{i:02d}' for i in range(1, 17)]
BEHAVIORAL_COLS = ['skor_motivasi', 'skor_kedisiplinan', 'skor_ekstrakurikuler',
                   'indeks_kehadiran', 'skor_literasi', 'skor_minat_belajar']
EXAM_COLS = ['kelas', 'urutan_ujian', 'skor_tryout']
TASK_COLS = ['tugas_selesai', 'tugas_diberikan']
DEMO_COLS = ['jarak_rumah_km', 'jumlah_saudara']


def _safe_div(a, b, eps=1e-6):
    return a / (b + eps)


def _clip_series(s, lower=0.01, upper=0.99):
    """Clip extreme values at percentiles."""
    lo, hi = s.quantile(lower), s.quantile(upper)
    return s.clip(lo, hi)


def engineer_features(df, is_train=True):
    """
    Add engineered features to dataframe.
    Returns new dataframe with additional columns.
    """
    df = df.copy()
    features = {}

    # ==========================================
    # A. WEEKLY SCORE FEATURES
    # ==========================================
    week = df[WEEK_COLS].values

    features['week_mean'] = np.nanmean(week, axis=1)
    features['week_median'] = np.nanmedian(week, axis=1)
    features['week_std'] = np.nanstd(week, axis=1)
    features['week_min'] = np.nanmin(week, axis=1)
    features['week_max'] = np.nanmax(week, axis=1)
    features['week_range'] = features['week_max'] - features['week_min']
    features['week_sum'] = np.nansum(week, axis=1)
    features['week_cv'] = _safe_div(features['week_std'], np.abs(features['week_mean']))

    # Quartiles & IQR
    features['week_q25'] = np.nanpercentile(week, 25, axis=1)
    features['week_q75'] = np.nanpercentile(week, 75, axis=1)
    features['week_iqr'] = features['week_q75'] - features['week_q25']

    # First / last
    features['week_first'] = week[:, 0]
    features['week_last'] = week[:, -1]
    features['week_last_minus_first'] = week[:, -1] - week[:, 0]

    # Trend features via simple linear regression per row
    x = np.arange(12)
    slopes = np.zeros(len(df))
    intercepts = np.zeros(len(df))
    r2s = np.zeros(len(df))
    for i in range(len(df)):
        y = week[i]
        mask = ~np.isnan(y)
        if mask.sum() >= 2:
            slope, intercept, r_val, _, _ = stats.linregress(x[mask], y[mask])
            slopes[i] = slope
            intercepts[i] = intercept
            r2s[i] = r_val ** 2
        else:
            slopes[i] = 0
            intercepts[i] = y[~np.isnan(y)].mean() if mask.sum() > 0 else 0
            r2s[i] = 0

    features['week_slope'] = slopes
    features['week_intercept'] = intercepts
    features['week_r2'] = r2s
    features['week_trend_strength'] = np.abs(slopes) * r2s

    # Early vs late
    early = week[:, :6]
    late = week[:, 6:]
    features['week_early_mean'] = np.nanmean(early, axis=1)
    features['week_late_mean'] = np.nanmean(late, axis=1)
    features['week_early_late_gap'] = features['week_early_mean'] - features['week_late_mean']

    # Changes
    diffs = np.diff(week, axis=1)
    features['week_pos_change_count'] = np.sum(diffs > 0, axis=1)
    features['week_neg_change_count'] = np.sum(diffs < 0, axis=1)
    features['week_zero_change_count'] = np.sum(np.abs(diffs) < 0.01, axis=1)
    features['week_mean_abs_change'] = np.nanmean(np.abs(diffs), axis=1)
    features['week_max_increase'] = np.max(diffs, axis=1)
    features['week_max_decrease'] = np.min(diffs, axis=1)

    # Outlier count (beyond 2 std from personal mean)
    centered = np.abs(week - features['week_mean'].reshape(-1, 1))
    features['week_outlier_count'] = np.sum(centered > 2 * features['week_std'].reshape(-1, 1) + 0.01, axis=1)

    # Rolling features
    # Simple: first 4 weeks mean vs last 4 weeks mean
    features['week_first4_mean'] = np.nanmean(week[:, :4], axis=1)
    features['week_mid4_mean'] = np.nanmean(week[:, 4:8], axis=1)
    features['week_last4_mean'] = np.nanmean(week[:, 8:], axis=1)
    features['week_first4_last4_diff'] = features['week_first4_mean'] - features['week_last4_mean']

    # ==========================================
    # B. ACTIVITY FEATURES
    # ==========================================
    act = df[ACTIVITY_COLS].values

    features['activity_mean'] = np.nanmean(act, axis=1)
    features['activity_median'] = np.nanmedian(act, axis=1)
    features['activity_std'] = np.nanstd(act, axis=1)
    features['activity_min'] = np.nanmin(act, axis=1)
    features['activity_max'] = np.nanmax(act, axis=1)
    features['activity_range'] = features['activity_max'] - features['activity_min']
    features['activity_sum'] = np.nansum(act, axis=1)
    features['activity_iqr'] = np.nanpercentile(act, 75, axis=1) - np.nanpercentile(act, 25, axis=1)

    # Activity slope
    x_act = np.arange(16)
    act_slopes = np.zeros(len(df))
    for i in range(len(df)):
        y = act[i]
        mask = ~np.isnan(y)
        if mask.sum() >= 2:
            act_slopes[i] = stats.linregress(x_act[mask], y[mask])[0]
    features['activity_slope'] = act_slopes

    features['activity_last_minus_first'] = act[:, -1] - act[:, 0]

    # Early vs late activity
    features['activity_early_mean'] = np.nanmean(act[:, :8], axis=1)
    features['activity_late_mean'] = np.nanmean(act[:, 8:], axis=1)
    features['activity_early_late_gap'] = features['activity_early_mean'] - features['activity_late_mean']

    # Activity patterns
    features['activity_zero_count'] = np.sum(np.abs(act) < 0.5, axis=1)
    features['activity_active_count'] = np.sum(act > 50, axis=1)
    features['activity_high_count'] = np.sum(act > 75, axis=1)
    features['activity_peak_ratio'] = _safe_div(np.sum(act > 70, axis=1).astype(float), 16)

    # Activity consistency (inverse of std)
    features['activity_consistency'] = _safe_div(1.0, features['activity_std'] + 0.01)

    # Best consecutive streak of high activity
    high = (act > 60).astype(int)
    streak_features = []
    for i in range(len(df)):
        # Find longest consecutive streak
        max_streak = 0
        current = 0
        for j in range(16):
            if high[i, j]:
                current += 1
                max_streak = max(max_streak, current)
            else:
                current = 0
        streak_features.append(max_streak)
    features['activity_max_streak'] = np.array(streak_features)

    # ==========================================
    # C. TASK FEATURES
    # ==========================================
    tugas_selesai = df['tugas_selesai'].values.astype(float)
    tugas_diberikan = df['tugas_diberikan'].values.astype(float)

    features['task_completion_ratio'] = _safe_div(tugas_selesai, tugas_diberikan)
    features['task_remaining'] = tugas_diberikan - tugas_selesai
    features['task_surplus'] = tugas_selesai - tugas_diberikan
    features['task_log_given'] = np.log1p(tugas_diberikan)
    features['task_log_completed'] = np.log1p(tugas_selesai)
    features['task_efficiency'] = _safe_div(tugas_selesai, tugas_diberikan + 1)
    features['task_per_week'] = _safe_div(tugas_diberikan, 12)
    features['task_completed_per_week'] = _safe_div(tugas_selesai, 12)

    # ==========================================
    # D. CROSS-GROUP INTERACTIONS
    # ==========================================
    week_mean = features['week_mean']
    week_std = features['week_std']
    act_mean = features['activity_mean']
    act_std = features['activity_std']
    task_ratio = features['task_completion_ratio']

    features['week_act_mean_prod'] = week_mean * act_mean
    features['week_act_mean_ratio'] = _safe_div(week_mean, act_mean + 0.1)
    features['week_act_std_prod'] = week_std * act_std
    features['task_week_prod'] = task_ratio * week_mean
    features['task_act_prod'] = task_ratio * act_mean

    # Tryout interactions
    tryout = df['skor_tryout'].values.astype(float)
    features['tryout_week_diff'] = tryout - week_mean
    features['tryout_week_prod'] = tryout * week_mean
    features['tryout_act_prod'] = tryout * act_mean
    features['tryout_week_ratio'] = _safe_div(tryout, week_mean + 0.1)

    # Behavioral interactions
    motivasi = df['skor_motivasi'].values.astype(float)
    disiplin = df['skor_kedisiplinan'].values.astype(float)
    features['motivasi_disiplin_prod'] = motivasi * disiplin
    features['motivasi_disiplin_sum'] = motivasi + disiplin
    features['motivasi_disiplin_diff'] = motivasi - disiplin

    # Consistency * performance
    features['consistency_perf'] = _safe_div(week_mean, week_std + 0.01) * (task_ratio + 0.1)

    # Trend * task completion
    features['trend_task'] = slopes * task_ratio

    # ==========================================
    # E. RATIO & COMPOSITE
    # ==========================================
    features['early_activity_vs_week'] = _safe_div(features['activity_early_mean'],
                                                   features['week_mean'] + 0.1)
    features['late_activity_vs_week'] = _safe_div(features['activity_late_mean'],
                                                  features['week_mean'] + 0.1)

    # Score volatility vs level
    features['week_volatility_ratio'] = _safe_div(week_std, np.abs(week_mean) + 0.1)

    # Overall "engagement" score
    features['engagement_score'] = (
        features['activity_mean'] / 100 +
        task_ratio +
        features['week_early_late_gap'].clip(-5, 5) / 10
    ) / 3

    # ==========================================
    # Combine
    # ==========================================
    feat_df = pd.DataFrame(features, index=df.index)

    # Clip extreme values
    for col in feat_df.select_dtypes(include=[np.number]).columns:
        if feat_df[col].dtype in [np.float64, np.float32]:
            feat_df[col] = _clip_series(feat_df[col])

    # Replace inf with nan, then fill
    feat_df = feat_df.replace([np.inf, -np.inf], np.nan).fillna(0)

    return feat_df


def get_all_features(df):
    """Return both original (minus id/target) and engineered features."""
    df = df.copy()
    drop_cols = ['id']
    if 'target' in df.columns:
        drop_cols.append('target')

    original = df.drop(columns=drop_cols)
    engineered = engineer_features(df)
    combined = pd.concat([original, engineered], axis=1)
    return combined
