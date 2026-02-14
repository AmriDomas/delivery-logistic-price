import pandas as pd

def detect_drift(baseline, request_buffer, threshold=0.15):
    if len(request_buffer) < 30:
        return {
            "status": "insufficient_data",
            "message": "Not enough requests for drift detection"
        }

    current_df = pd.DataFrame(request_buffer)

    drift_report = {}
    drift_detected = False

    for feature, base_mean in baseline["feature_means"].items():
        if feature in current_df.columns:
            curr_mean = current_df[feature].mean()
            diff_ratio = abs(curr_mean - base_mean) / (abs(base_mean) + 1e-6)
            is_drift = diff_ratio > threshold

            drift_report[feature] = {
                "baseline_mean": float(base_mean),
                "current_mean": float(curr_mean),
                "diff_ratio": float(diff_ratio),
                "drift": is_drift
            }

            if is_drift:
                drift_detected = True

    return {
        "status": "ok",
        "drift_detected": drift_detected,
        "details": drift_report
    }
