def select_best_model(results):
    return max(results.items(), key=lambda x: x[1]['accuracy'])
 
