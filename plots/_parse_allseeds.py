import re

def parse_final_results_allseeds(path):
    with open(path, 'r', encoding='utf-8') as f:
        text = f.read()
    blocks = text.split('Loaded 400 queries with complete records')[1:]
    seeds = [42, 7, 123, 2024]
    results = []
    for seed, block in zip(seeds, blocks):
        static_p95, static_mean, static_faith = re.search('Static baseline:\\s+P95\\s+([\\d.]+)\\s+s\\s+mean\\s+([\\d.]+)\\s+s\\s+faithfulness\\s+([\\d.]+)', block).groups()
        adapt_p95, adapt_mean, adapt_faith = re.search('Adaptive \\(K=3\\):\\s+P95\\s+([\\d.]+)\\s+s\\s+mean\\s+([\\d.]+)\\s+s\\s+faithfulness\\s+([\\d.]+)', block).groups()
        p95_pct, mean_pct, faith_delta = re.search('Change:\\s+P95\\s+(-?[\\d.]+)%\\s+mean\\s+(-?[\\d.]+)%\\s+faithfulness\\s+(-?[\\d.]+)', block).groups()
        fast_n, deep_n = re.search('Fast/Deep split:\\s+(\\d+)\\s+fast\\s+/\\s+(\\d+)\\s+deep', block).groups()
        lost, gained, p_value = re.search('McNemar on faithfulness:\\s+lost\\s+(\\d+),\\s+gained\\s+(\\d+),\\s+p\\s*=\\s*([\\d.]+)', block).groups()
        fast_rate, deep_rate = re.search('Deep-needed rate among Fast-routed:\\s+([\\d.]+)%\\s+among Deep-routed:\\s+([\\d.]+)%', block).groups()
        results.append({'seed': seed, 'static_p95': float(static_p95), 'static_mean': float(static_mean), 'static_faith': float(static_faith), 'adaptive_p95': float(adapt_p95), 'adaptive_mean': float(adapt_mean), 'adaptive_faith': float(adapt_faith), 'p95_pct_change': float(p95_pct), 'mean_pct_change': float(mean_pct), 'faith_delta': float(faith_delta), 'fast_n': int(fast_n), 'deep_n': int(deep_n), 'mcnemar_lost': int(lost), 'mcnemar_gained': int(gained), 'mcnemar_p': float(p_value), 'deep_needed_rate_fast_routed': float(fast_rate), 'deep_needed_rate_deep_routed': float(deep_rate)})
    return results
if __name__ == '__main__':
    import json
    r = parse_final_results_allseeds('../results/final/FINAL_results_allseeds.txt')
    print(json.dumps(r, indent=2))
