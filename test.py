import torch
from model import SimpleCNN_LSTM
from get_data import get_train_test_data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from sklearn.metrics import r2_score, mean_absolute_error

# 设置中文显示
rcParams['font.family'] = 'SimHei'
rcParams['axes.unicode_minus'] = False

# 功率段定义和标定效率
POWER_SEGMENTS = [
    (1000, 1200, 8.6),  # 功率范围, 标定效率%
    (1800, 2400, 14.6),
    (4700, 5400, 25.0),
    (3700, 4300, 22.8),
    (2800, 3200, 19.6)
]

# 柴油热值 (J/kg)
DIESEL_HEAT_VALUE = 42.6 * 10 ** 6

PATH = 'model/cnn-lstm.pkl'


def get_power_segment(power_w):
    """根据功率值判断所属功率段"""
    for min_power, max_power, baseline_eff in POWER_SEGMENTS:
        if min_power <= power_w < max_power:
            return f"{int((min_power + max_power) / 2)}W段", baseline_eff
    return "未知功率段", None


def calculate_power(oil_flow_g_per_s, efficiency_percent):
    """计算功率(W) = 油量(g/s) × 42600 × (效率%/100)"""
    return oil_flow_g_per_s * 42600 * (efficiency_percent / 100.0)


def main():
    _, test_loader = get_train_test_data('训练数据.csv', interval=10)
    dataset = test_loader.dataset

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SimpleCNN_LSTM(input_size=5, hidden_size=32, dropout=0.3)

    model.load_state_dict(torch.load(PATH, map_location=device))
    model.to(device)
    model.eval()

    predictions = []
    normalized_actuals = []
    all_features = []  # 保存原始特征值用于功率计算

    with torch.no_grad():
        for seq, target in test_loader:
            seq, target = seq.to(device), target.to(device)
            output = model(seq)
            predictions.extend(output.view(-1).cpu().numpy())
            normalized_actuals.extend(target.view(-1).cpu().numpy())

    # 反归一化
    predictions = np.array(predictions).reshape(-1, 1)
    normalized_actuals = np.array(normalized_actuals).reshape(-1, 1)

    actuals = dataset.target_scaler.inverse_transform(normalized_actuals).flatten()
    predictions = dataset.target_scaler.inverse_transform(predictions).flatten()

    # 获取原始特征值（第一列是油量）
    original_features = dataset.original_features[:len(actuals)]
    oil_flow_values = original_features[:, 0]  # 油量值 (g/s)

    results = []
    for i in range(len(actuals)):
        # 计算功率
        power_w = calculate_power(oil_flow_values[i], predictions[i])

        # 判断功率段和获取标定效率
        segment_name, baseline_eff = get_power_segment(power_w)

        if baseline_eff is not None:
            # 计算退化程度
            degradation = baseline_eff - predictions[i]
        else:
            degradation = np.nan

        results.append({
            'index': i,
            'actual_efficiency': actuals[i],
            'predicted_efficiency': predictions[i],
            'degradation': degradation,
            'power_segment': segment_name,
            'calculated_power': power_w,
            'oil_flow': oil_flow_values[i]
        })

    # 创建结果DataFrame
    df_results = pd.DataFrame(results)
    df_results.to_csv('efficiency_degradation_results.csv', index=False, encoding='utf-8-sig')

    # 计算评估指标
    mae = mean_absolute_error(df_results['actual_efficiency'], df_results['predicted_efficiency'])
    r2 = r2_score(df_results['actual_efficiency'], df_results['predicted_efficiency'])

    # 绘制结果图
    plt.figure(figsize=(15, 10))

    # 子图1: 效率对比和退化程度
    plt.subplot(2, 1, 1)
    x = range(len(df_results))

    # 效率对比
    plt.plot(x, df_results['actual_efficiency'], 'b-', label='效率实际值(%)', linewidth=2, marker='o', markersize=4)
    plt.plot(x, df_results['predicted_efficiency'], 'r-', label='效率预测值(%)', linewidth=2, marker='x', markersize=4)

    # 退化程度（柱状图）
    plt.bar(x, df_results['degradation'], alpha=0.3, color='green', label='效率退化程度(%)')

    plt.xlabel('数据点索引')
    plt.ylabel('效率值(%) / 退化程度(%)')
    plt.title(f'效率预测与退化分析\nMAE: {mae:.3f}%, R²: {r2:.4f}')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 子图2: 功率段分布
    plt.subplot(2, 1, 2)

    # 为每个功率段设置不同颜色
    colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow', 'lightpink']
    segment_colors = {}

    for idx, (min_power, max_power, baseline_eff) in enumerate(POWER_SEGMENTS):
        segment_name = f"{int((min_power + max_power) / 2)}W段"
        segment_colors[segment_name] = colors[idx % len(colors)]

    # 绘制功率段背景
    current_segment = None
    start_idx = 0

    for i, segment in enumerate(df_results['power_segment']):
        if segment != current_segment:
            if current_segment is not None:
                plt.axvspan(start_idx, i - 1, alpha=0.3, color=segment_colors.get(current_segment, 'gray'))
            current_segment = segment
            start_idx = i
    plt.axvspan(start_idx, len(df_results) - 1, alpha=0.3, color=segment_colors.get(current_segment, 'gray'))

    # 绘制功率值
    plt.plot(x, df_results['calculated_power'] / 1000, 'purple', label='计算功率(kW)', linewidth=2)
    plt.xlabel('数据点索引')
    plt.ylabel('功率(kW)')
    plt.title('功率段分布')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 添加功率段图例
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=color, alpha=0.3, label=segment)
                       for segment, color in segment_colors.items()]
    plt.legend(handles=legend_elements, loc='upper right')

    plt.tight_layout()
    plt.savefig('efficiency_degradation_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 打印统计信息
    print(f"\n=== 测试结果统计 ===")
    print(f"平均绝对误差(MAE): {mae:.3f}%")
    print(f"决定系数(R²): {r2:.4f}")
    print(f"平均退化程度: {df_results['degradation'].mean():.3f}%")
    print(f"最大退化程度: {df_results['degradation'].max():.3f}%")
    print(f"最小退化程度: {df_results['degradation'].min():.3f}%")

    print(f"\n=== 各功率段统计 ===")
    for segment in df_results['power_segment'].unique():
        segment_data = df_results[df_results['power_segment'] == segment]
        print(f"{segment}: {len(segment_data)}个样本, "
              f"平均退化: {segment_data['degradation'].mean():.3f}%")


if __name__ == '__main__':
    main()