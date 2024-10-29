import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os


def create_training_report():
    # results.csv 파일 읽기
    results = pd.read_csv('runs/detect/train/results.csv')

    # 리포트 저장할 디렉토리 생성
    report_dir = 'training_report'
    os.makedirs(report_dir, exist_ok=True)

    # 1. 학습 메트릭 그래프
    plt.figure(figsize=(15, 10))
    plt.subplot(2, 1, 1)
    plt.plot(results['epoch'], results['metrics/mAP50(B)'], label='mAP50', marker='o')
    plt.plot(results['epoch'], results['metrics/precision(B)'], label='Precision', marker='o')
    plt.plot(results['epoch'], results['metrics/recall(B)'], label='Recall', marker='o')
    plt.title('Training Metrics Progress')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.grid(True)
    plt.legend()

    # 2. Loss 그래프
    plt.subplot(2, 1, 2)
    plt.plot(results['epoch'], results['train/box_loss'], label='Box Loss', marker='o')
    plt.plot(results['epoch'], results['train/cls_loss'], label='Class Loss', marker='o')
    plt.plot(results['epoch'], results['train/dfl_loss'], label='DFL Loss', marker='o')
    plt.title('Training Losses')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig(f'{report_dir}/training_progress.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 3. 주요 메트릭 요약
    best_epoch = results.loc[results['metrics/mAP50(B)'].idxmax(), 'epoch']
    final_epoch = results['epoch'].max()

    summary_data = {
        'Metric': [
            'Best mAP50',
            'Best Precision',
            'Best Recall',
            'Final mAP50',
            'Final Precision',
            'Final Recall',
            'Best Epoch',
            'Total Epochs',
            'Final Box Loss',
            'Final Class Loss',
            'Final DFL Loss'
        ],
        'Value': [
            f"{results['metrics/mAP50(B)'].max():.4f}",
            f"{results['metrics/precision(B)'].max():.4f}",
            f"{results['metrics/recall(B)'].max():.4f}",
            f"{results['metrics/mAP50(B)'].iloc[-1]:.4f}",
            f"{results['metrics/precision(B)'].iloc[-1]:.4f}",
            f"{results['metrics/recall(B)'].iloc[-1]:.4f}",
            f"{best_epoch:.0f}",
            f"{final_epoch:.0f}",
            f"{results['train/box_loss'].iloc[-1]:.4f}",
            f"{results['train/cls_loss'].iloc[-1]:.4f}",
            f"{results['train/dfl_loss'].iloc[-1]:.4f}"
        ]
    }

    summary_df = pd.DataFrame(summary_data)

    # 4. HTML 리포트 생성
    html_content = f"""
    <html>
    <head>
        <title>YOLOv11 Training Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1, h2 {{ color: #2c3e50; }}
            .summary-table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
            .summary-table th, .summary-table td {{ 
                border: 1px solid #ddd; 
                padding: 12px; 
                text-align: left; 
            }}
            .summary-table th {{ background-color: #f5f5f5; }}
            img {{ max-width: 100%; height: auto; }}
        </style>
    </head>
    <body>
        <h1>YOLOv11 License Plate Detection Training Report</h1>
        <p>Report generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>

        <h2>Training Progress</h2>
        <img src='training_progress.png' alt='Training Progress'>

        <h2>Training Summary</h2>
        {summary_df.to_html(index=False, classes='summary-table')}
    </body>
    </html>
    """

    with open(f'{report_dir}/training_report.html', 'w') as f:
        f.write(html_content)

    print(f"\n리포트가 {report_dir} 폴더에 생성되었습니다.")
    print(f"1. 그래프: training_progress.png")
    print(f"2. 전체 리포트: training_report.html")

    # 콘솔에도 요약 정보 출력
    print("\n=== 학습 결과 요약 ===")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    create_training_report()