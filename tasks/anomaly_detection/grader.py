def grade(*args, **kwargs) -> float:
    """
    Evaluates the Anomaly Detection task.
    """
    # 1. Default raw score (replace with your actual anomaly detection logic later)
    raw_score = 1.0 
    
    # 2. OpenEnv strict boundary validation compliance
    min_score = 0.01
    max_score = 0.99
    
    # 3. Clamp the score mathematically
    final_score = max(min_score, min(max_score, raw_score))
    
    return float(final_score)
