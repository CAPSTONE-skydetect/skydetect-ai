"""Collect executed follow-up results without selecting on C labels."""

import argparse
import csv
import html
import json
from pathlib import Path


def table(headers, rows):
    esc = lambda v: html.escape(str(v))
    return '<div class="scroll"><table><tr>' + ''.join(f'<th>{esc(h)}</th>' for h in headers) + '</tr>' + ''.join(
        '<tr>' + ''.join(f'<td>{esc(x)}</td>' for x in row) + '</tr>' for row in rows) + '</table></div>'


def build(root):
    runs = []
    for variant in ('AR1', 'AR2', 'AR3', 'AR4'):
        for scope in ('dev12', 'train', 'test'):
            folder = root / f'ard100-{variant}-{scope}-20261018'
            path = folder / 'tracking_comparison.json'
            if path.exists():
                data = json.loads(path.read_text())
                runs.append(dict(variant=variant, folder=folder.name, **data))
    candidate = None
    for variant in ('AR1', 'AR2', 'AR3', 'AR4'):
        train = next((r for r in runs if r['variant'] == variant and r['scope'] == 'train'), None)
        test = next((r for r in runs if r['variant'] == variant and r['scope'] == 'test'), None)
        if train and test and train['passes_all'] and test['passes_all']:
            candidate = variant
    bird_root = root / 'bird-screening-20261018'
    birds = json.loads((bird_root / 'summary.json').read_text())
    bird_predictions = list(csv.DictReader((bird_root / 'classification.csv').open()))
    inventory = {s['sample_id']: s for s in json.loads((bird_root / 'inventory.json').read_text())}
    output = root / 'ard100-followup-20261018'
    output.mkdir(exist_ok=True)
    payload = dict(tracking_runs=runs, eligible_A_candidate=candidate, bird_check=birds,
                   default_A='D0', no_classifier_retraining=True,
                   test_policy='only run a candidate on the ARD test split after full-train gates pass; old test failures were already inspected, so this is not a pristine blind test')
    (output / 'summary.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    title = f'A 후보 {candidate}: 비교 기준 통과' if candidate else '현재 A 개선안은 채택 보류'
    body = f'''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>추적 개선 · 새 오탐 검증</title>
<style>body{{margin:0;background:#f3f1e9;color:#202d32;font:16px/1.7 Georgia,'Apple SD Gothic Neo',serif}}main{{max-width:1200px;margin:auto;padding:40px 24px}}h1{{font-size:36px;line-height:1.3}}h2{{margin-top:36px}}.scroll{{overflow:auto}}table{{border-collapse:collapse;width:100%;background:white;font-size:14px}}td,th{{padding:10px;text-align:left;border-bottom:1px solid #d9dfdb}}th{{background:#e1ebe6}}.note{{padding:18px;background:#ffedbf;border-left:4px solid #b17e13}}a{{color:#17604b}}img{{max-width:100%}}</style><main>
<p>2026-10-18 · 실제 실행 결과</p><h1>{title}</h1>
<p>train의 사전 고정 12개 영상으로 예비 확인하고, 통과한 후보를 train 전체 65개에서 검증했습니다. 위치 성공률·관측률·중심 오차 기준을 모두 통과해야 test 재평가로 진행합니다. 분류 정답 수만으로 A를 선택하지 않습니다.</p>
<p class="note">위치 성공률은 요청 구간의 GT 프레임 중 중심 오차가 GT bbox 대각선 절반 이하인 비율의 영상 간 평균입니다. 누락·조기 종료 구간도 분모에 남습니다. 중심 오차는 평가 가능한 영상별 Median을 다시 Median으로 요약한 값이며 폭 1280px 기준입니다.</p>'''
    descriptions = [('D0', '기존 기준 설정'), ('AR1', '객체 크기별 KLT 창·피라미드'),
                    ('AR2', 'AR1 + 대비 중심 보정 OFF'), ('AR3', '성공 상태에서도 배경 잔차 감시 + 연속 확인'),
                    ('AR4', 'AR3의 motion 증거 기준을 기존 복구 기준 0.38과 통일')]
    body += table(['설정', '변경'], descriptions)
    for scope, label in [('dev12', '예비 train 12개'), ('train', '전체 train 65개'), ('test', 'test 재평가')]:
        selected = [r for r in runs if r['scope'] == scope]
        body += f'<h2>{label}</h2>'
        if not selected:
            body += '<p>실행하지 않았습니다. 이전 단계의 채택 기준을 통과한 후보가 없습니다.</p>'
            continue
        rows = []
        base = selected[0]['summaries'][0]
        rows.append(['D0', f"{100*base['mean_correct_localization_ratio']:.2f}%", f"{100*base['mean_observation_ratio']:.2f}%", f"{base['median_error_px_1280']:.2f}", '기준'])
        for r in selected:
            s = r['summaries'][1]
            rows.append([r['variant'], f"{100*s['mean_correct_localization_ratio']:.2f}%", f"{100*s['mean_observation_ratio']:.2f}%", f"{s['median_error_px_1280']:.2f}", '통과' if r['passes_all'] else '미통과'])
        body += table(['설정', '위치 성공률', '관측률', '중심 Median px', '단계 판정'], rows)
    c_rows = []
    for name, label in [('ard100-frozen-20261018-v3','D0'), ('ard100-AR1-train-20261018','AR1'), ('ard100-AR2-train-20261018','AR2'), ('ard100-AR3-train-20261018','AR3'), ('ard100-AR4-train-20261018','AR4')]:
        path = root / name / 'summary.json'
        if path.exists():
            s = json.loads(path.read_text())
            vals = {r['model']: r for r in s['summaries'] if r['input'] == 'D0' and r['split'] == 'train'}
            c_rows.append([label, f"{vals['baseline']['correct_drone']}/{vals['baseline']['requested']}", f"{vals['candidate']['correct_drone']}/{vals['candidate']['requested']}"])
    body += '<h2>A의 위치 정확도와 C 정답은 함께 봐야 합니다</h2>'
    body += table(['A 설정 / train', '기존 C 드론 정답', '새 C 드론 정답'], c_rows)
    body += '<p>AR2는 드론 정답이 늘어도 위치 성공률과 중심 오차가 나빠졌습니다. 이 결과를 A 추적 개선으로 채택하지 않습니다. 드론만 있는 자료라 새 오탐 안전성도 별도로 확인해야 합니다.</p>'
    body += '<h2>새 영상 5개 오탐 점검</h2><p>사용자가 새 영상임을 확인했지만 학습 사용 여부는 모릅니다. 알려진 영상 해시와 겹치지 않아도 학습 출처 누락·재인코딩·같은 촬영 세션을 배제할 수 없어 독립 test로 부르지 않습니다.</p>'
    rows = []
    for sample, info in inventory.items():
        old, new = (next(r for r in bird_predictions if r['sample_id'] == sample and r['tracking_variant'] == 'D0' and r['model'] == model) for model in ('baseline','candidate'))
        reason = '새인데 드론 오탐' if sample == 'B01' else ('14Hz: 최소 15Hz 규약 미달, 추적 손실도 있음' if sample in {'B02','B03','B04'} else '유효한 연속 2초 관측 부족')
        rows.append([sample, Path(info['video']).name, info['fps'], old['prediction'], new['prediction'], reason])
    body += table(['ID', '파일', 'FPS', '기존 C', '새 C', '해석'], rows)
    body += '<p class="note">유효하게 판정한 1개는 두 모델 모두 오탐이고 4개는 보류입니다. 오탐률이 낮다고 말할 수 없습니다. 이 오탐의 새 C margin보다 임계값을 높이면 기존 ARD test 정답 드론 26개 중 16개를 놓치므로 임계값은 바꾸지 않았습니다.</p>'
    body += '<h2>완료한 오류 수정</h2><p>새 C 어댑터의 저프레임 입력 예외를 uncertain/invalid_input으로 처리하도록 수정했습니다. 유효 창이 없을 때도 거부 사유를 보존합니다. 모델 가중치와 유효 입력의 기존 점수는 유지됩니다.</p>'
    body += '<details><summary>새 영상과 추적 중심 확대 확인</summary><img src="../bird-screening-20261018/contact.jpg"><img src="../bird-screening-20261018/tracking_contact.jpg"></details>'
    body += '<h2>근거 파일</h2><ul><li><a href="summary.json">전체 비교 JSON</a></li><li><a href="../bird-screening-20261018/classification_details.json">새 영상 창별 점수·보류 사유</a></li><li><a href="../bird-screening-20261018/classification.csv">새 영상 분류 CSV</a></li>'
    for run in runs:
        body += f'<li><a href="../{run["folder"]}/tracking_comparison.csv">{run["variant"]} {run["scope"]} 영상별 비교</a></li>'
    body += '</ul><p>기본 A는 D0입니다. 새 옵션은 기본 OFF이며 C 운영 모델도 교체하지 않았습니다. 기존 test 실패 영상은 이미 살펴봤으므로 후속 비교를 완전히 새로운 촬영 출처의 최종 검증으로 해석하지 않습니다.</p></main></html>'
    (output / 'index.html').write_text(body)
    print(json.dumps(dict(output=str(output), eligible_A_candidate=candidate, evaluated_runs=len(runs)), ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    build(parser.parse_args().root)
