-- ============================================================================
-- Heart Failure Death Event Prediction — SQL Queries
-- Author: Neha Tiwari
-- Database: clinical_data  |  Table: heart_failure_patients
-- Purpose: Clinical insight queries for patient risk profiling
-- ============================================================================


-- ── Q1: Overall mortality summary ────────────────────────────────────────────
SELECT
    COUNT(*)                                                          AS total_patients,
    SUM(death_event)                                                  AS deaths,
    COUNT(*) - SUM(death_event)                                       AS survived,
    ROUND(100.0 * SUM(death_event) / COUNT(*), 2)                    AS mortality_rate_pct,
    ROUND(AVG(age), 1)                                                AS avg_age,
    ROUND(AVG(ejection_fraction), 1)                                  AS avg_ejection_fraction,
    ROUND(AVG(serum_creatinine), 2)                                   AS avg_serum_creatinine
FROM heart_failure_patients;


-- ── Q2: Mortality by age group ────────────────────────────────────────────────
SELECT
    CASE
        WHEN age BETWEEN 40 AND 50 THEN '40–50'
        WHEN age BETWEEN 51 AND 60 THEN '51–60'
        WHEN age BETWEEN 61 AND 70 THEN '61–70'
        WHEN age BETWEEN 71 AND 80 THEN '71–80'
        ELSE '81+'
    END                                                               AS age_group,
    COUNT(*)                                                          AS total_patients,
    SUM(death_event)                                                  AS deaths,
    ROUND(100.0 * SUM(death_event) / COUNT(*), 2)                    AS mortality_rate_pct,
    ROUND(AVG(ejection_fraction), 1)                                  AS avg_ejection_fraction,
    ROUND(AVG(serum_creatinine), 2)                                   AS avg_serum_creatinine
FROM heart_failure_patients
GROUP BY age_group
ORDER BY MIN(age);


-- ── Q3: Mortality by clinical risk factors ────────────────────────────────────
-- Compare mortality across binary conditions
SELECT
    'Anaemia'             AS risk_factor,
    SUM(CASE WHEN anaemia = 1 AND death_event = 1 THEN 1 ELSE 0 END) AS deaths_with_condition,
    SUM(CASE WHEN anaemia = 1 THEN 1 ELSE 0 END)                      AS patients_with_condition,
    ROUND(100.0 * SUM(CASE WHEN anaemia = 1 AND death_event = 1 THEN 1 ELSE 0 END)
          / NULLIF(SUM(CASE WHEN anaemia = 1 THEN 1 ELSE 0 END), 0), 2) AS mortality_rate_pct
FROM heart_failure_patients

UNION ALL

SELECT
    'High Blood Pressure',
    SUM(CASE WHEN high_blood_pressure = 1 AND death_event = 1 THEN 1 ELSE 0 END),
    SUM(CASE WHEN high_blood_pressure = 1 THEN 1 ELSE 0 END),
    ROUND(100.0 * SUM(CASE WHEN high_blood_pressure = 1 AND death_event = 1 THEN 1 ELSE 0 END)
          / NULLIF(SUM(CASE WHEN high_blood_pressure = 1 THEN 1 ELSE 0 END), 0), 2)
FROM heart_failure_patients

UNION ALL

SELECT
    'Diabetes',
    SUM(CASE WHEN diabetes = 1 AND death_event = 1 THEN 1 ELSE 0 END),
    SUM(CASE WHEN diabetes = 1 THEN 1 ELSE 0 END),
    ROUND(100.0 * SUM(CASE WHEN diabetes = 1 AND death_event = 1 THEN 1 ELSE 0 END)
          / NULLIF(SUM(CASE WHEN diabetes = 1 THEN 1 ELSE 0 END), 0), 2)
FROM heart_failure_patients

UNION ALL

SELECT
    'Smoking',
    SUM(CASE WHEN smoking = 1 AND death_event = 1 THEN 1 ELSE 0 END),
    SUM(CASE WHEN smoking = 1 THEN 1 ELSE 0 END),
    ROUND(100.0 * SUM(CASE WHEN smoking = 1 AND death_event = 1 THEN 1 ELSE 0 END)
          / NULLIF(SUM(CASE WHEN smoking = 1 THEN 1 ELSE 0 END), 0), 2)
FROM heart_failure_patients

ORDER BY mortality_rate_pct DESC;


-- ── Q4: Ejection fraction bands and mortality ─────────────────────────────────
-- Clinical thresholds: <40% = low, 40-55% = borderline, >55% = normal
SELECT
    CASE
        WHEN ejection_fraction < 30  THEN 'Severely Low (<30%)'
        WHEN ejection_fraction < 40  THEN 'Low (30–39%)'
        WHEN ejection_fraction < 55  THEN 'Borderline (40–54%)'
        ELSE 'Normal (55%+)'
    END                                                               AS ef_band,
    COUNT(*)                                                          AS total_patients,
    SUM(death_event)                                                  AS deaths,
    ROUND(100.0 * SUM(death_event) / COUNT(*), 2)                    AS mortality_rate_pct,
    ROUND(AVG(serum_creatinine), 2)                                   AS avg_serum_creatinine
FROM heart_failure_patients
GROUP BY ef_band
ORDER BY MIN(ejection_fraction);


-- ── Q5: Serum creatinine risk bands and mortality ─────────────────────────────
-- Clinical normal range: 0.6–1.2 mg/dL for women, 0.7–1.3 mg/dL for men
SELECT
    CASE
        WHEN serum_creatinine <= 1.2 THEN 'Normal (≤1.2)'
        WHEN serum_creatinine <= 2.0 THEN 'Mildly Elevated (1.3–2.0)'
        WHEN serum_creatinine <= 4.0 THEN 'Moderately Elevated (2.1–4.0)'
        ELSE 'Severely Elevated (>4.0)'
    END                                                               AS creatinine_band,
    COUNT(*)                                                          AS total_patients,
    SUM(death_event)                                                  AS deaths,
    ROUND(100.0 * SUM(death_event) / COUNT(*), 2)                    AS mortality_rate_pct
FROM heart_failure_patients
GROUP BY creatinine_band
ORDER BY MIN(serum_creatinine);


-- ── Q6: High-risk patient segment ────────────────────────────────────────────
-- Patients with multiple critical risk markers — still alive (intervention targets)
SELECT
    age,
    ejection_fraction,
    serum_creatinine,
    serum_sodium,
    high_blood_pressure,
    anaemia,
    time                                                              AS followup_days
FROM heart_failure_patients
WHERE
    death_event = 0                         -- Still alive
    AND ejection_fraction < 35              -- Low heart pumping efficiency
    AND serum_creatinine > 1.5             -- Elevated kidney marker
    AND age > 65                            -- Older patient
ORDER BY serum_creatinine DESC;


-- ── Q7: Follow-up period vs mortality ────────────────────────────────────────
-- Shorter follow-up correlates with death — patients lost to follow-up die sooner
SELECT
    CASE
        WHEN time <= 30  THEN '0–30 days'
        WHEN time <= 90  THEN '31–90 days'
        WHEN time <= 180 THEN '91–180 days'
        ELSE '181+ days'
    END                                                               AS followup_band,
    COUNT(*)                                                          AS total_patients,
    SUM(death_event)                                                  AS deaths,
    ROUND(100.0 * SUM(death_event) / COUNT(*), 2)                    AS mortality_rate_pct
FROM heart_failure_patients
GROUP BY followup_band
ORDER BY MIN(time);


-- ── Q8: Gender-based clinical profile ────────────────────────────────────────
SELECT
    CASE sex WHEN 0 THEN 'Female' ELSE 'Male' END                    AS gender,
    COUNT(*)                                                          AS total,
    SUM(death_event)                                                  AS deaths,
    ROUND(100.0 * SUM(death_event) / COUNT(*), 2)                    AS mortality_rate_pct,
    ROUND(AVG(age), 1)                                                AS avg_age,
    ROUND(AVG(ejection_fraction), 1)                                  AS avg_ejection_pct,
    ROUND(AVG(serum_creatinine), 2)                                   AS avg_creatinine,
    SUM(smoking)                                                      AS smokers
FROM heart_failure_patients
GROUP BY sex;


-- ── Q9: Multi-condition co-morbidity and mortality ───────────────────────────
SELECT
    (anaemia + diabetes + high_blood_pressure + smoking)              AS comorbidity_count,
    COUNT(*)                                                          AS patients,
    SUM(death_event)                                                  AS deaths,
    ROUND(100.0 * SUM(death_event) / COUNT(*), 2)                    AS mortality_rate_pct
FROM heart_failure_patients
GROUP BY comorbidity_count
ORDER BY comorbidity_count;


-- ── Q10: Combined high-risk flag — critical patient dashboard ─────────────────
SELECT
    COUNT(*)                                                           AS total_patients,
    SUM(CASE
            WHEN ejection_fraction < 35
             AND serum_creatinine > 1.5
             AND age > 65
            THEN 1 ELSE 0
        END)                                                           AS critical_risk_patients,
    ROUND(100.0 * SUM(CASE
                          WHEN ejection_fraction < 35
                           AND serum_creatinine > 1.5
                           AND age > 65
                          THEN 1 ELSE 0
                      END) / COUNT(*), 2)                              AS critical_risk_pct,
    ROUND(100.0 * SUM(CASE
                          WHEN ejection_fraction < 35
                           AND serum_creatinine > 1.5
                           AND age > 65
                           AND death_event = 1
                          THEN 1 ELSE 0
                      END)
          / NULLIF(SUM(CASE
                           WHEN ejection_fraction < 35
                            AND serum_creatinine > 1.5
                            AND age > 65
                           THEN 1 ELSE 0
                       END), 0), 2)                                    AS critical_mortality_pct
FROM heart_failure_patients;
