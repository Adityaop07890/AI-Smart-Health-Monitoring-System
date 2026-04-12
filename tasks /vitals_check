def grade(completion, state=None, **kwargs) -> float:
    # 1. Your custom logic to calculate the score goes here
    raw_score = 0.0 
    
    # Example logic: if the agent successfully completes the task, it gets a 1.0
    if completion and len(str(completion)) > 0:
        raw_score = 1.0
    else:
        raw_score = 0.0
        
    # 2. CRITICAL FIX: Clamp the score to be strictly between 0 and 1
    # This prevents the autograder from silently dropping the task
    min_score = 0.01
    max_score = 0.99
    
    final_score = max(min_score, min(max_score, raw_score))
    
    # 3. Return the clamped float
    return float(final_score)
