
# -*- coding: utf-8 -*-
"""[시간별 최종 단계] APN / Eng'r 판정 → APN_LOT.csv append + 리포트 데이터 생성.

1) 시간별 추천 파일 (data/recommend/recommend_future_*.csv) 규칙:
   - 추천행의 anchor = latest_block_no (없으면 latest_wire의 마지막 블럭).
   - parquet에서 anchor 블럭을 동일 EQP_ID + process_time 시간순에서 찾은 뒤,
     그 블럭의 **다음다음 run (idx+2)** 을 판정 대상 블럭으로 삼는다
     (lead = "다음다음 lot" - 추천은 그 run에 이 recipe로 들어가라는 의미)
   - 대상 블럭의 frame 12지점 프로파일 vs 추천 rec_set_frame_temp_*:
     장비 setpoint는 소수 1째자리(0.1°C) 판별 → 반올림 후 12지점 전부 동일하면 frame_match
   - slurry 동일 기준 → slurry_match
   - **frame OR slurry** match → `APN`, 둘 다 아니면 → `Eng'r 변경`
   - APN_LOT.csv(LOT_ID,Tuning_Group,EQP_ID)에 append - (LOT_ID,EQP_ID) 중복은
     기존 행 유지 (수동 작성 APN lot 9개 등 우선). 대상 블럭이 아직 parquet에 없으면 대기.

2) 리포트 데이터 (check_result/apn_eng_data/apn_eng_blocks.csv, 매 run 재생성):
   - **EQP_NM 대상 필터 (EQUIP_FILTER 31개 pilot 장비)** - 그 외 장비는 CSV에서 제외
   - APN_LOT.csv 의 모든 lot에 대해 실제 bow / 레시피 / 모델 예측을 붙인다
     - pilot_results.csv(2026-09-22 pilot 45블럭)에 있으면 그 값 사용
     - 없으면 parquet에서 블럭 역산(back) - 직전 10wire roll 조건으로
       recommend_future.inverse_profile 재현 (역산 드리프트 1스텝 0.1°C 허용)
   - APN(추천 레시피 직접 투입) : pred_rec(약속 bow) vs actual_bow = err_rec
   - Eng'r(추천 미투입)          : pred_actual(실제 recipe forward) vs actual_bow = err_actual
   → 10_plot_apn_eng.py 가 group_compare.html 생성

사용법:
    python compare_apn_eng.py
"""
import glob
import json
import os
import re
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd
import pyarrow.parquet as parq

_BASE = os.environ.get('BASE_DIR', r'D:\chaewon\APC\04.TF_APN')
REC_DIR = os.path.join(_BASE, 'data', 'recommend')
PQ_PATH = os.path.join(_BASE, 'data', 'WireSaw_Field_Test_preprocessed.parquet')
APN_LOT_CSV = os.path.join(_BASE, 'APN_LOT.csv')
PILOT_RESULTS = os.path.join(_BASE, 'check_result', 'pilot_data', 'pilot_results.csv')
DATA_DIR = os.path.join(_BASE, 'check_result', 'apn_eng_data')
BLOCKS_CSV = os.path.join(DATA_DIR, 'apn_eng_blocks.csv')
EQP_NM_CSV = os.path.join(_BASE, 'eqp_이름.csv')
DAILY_GLOB = os.path.join(_BASE, 'data', 'WireSaw_Field_Test_20*.parquet')
DAILY_RAW_COLS = ['EQP_ID', 'BLK_NO', 'TRACE_DTTS', 'SL-TTV-ALL', 'SL-THK-AV']
BY_EQP_DIR = os.path.join(_BASE, 'data', 'by_eqp')   # split_by_eqp.py 출력 (장비별 raw)
TTV_THK_CACHE = os.path.join(DATA_DIR, 'ttv_thk_by_block.parquet')
TTV_THK_MANIFEST = os.path.join(DATA_DIR, 'ttv_thk_manifest.json')

# ── 전체 히스토리 블록 프로필 ("Other" lot 보강용) ──
# apn_eng_blocks.csv는 08-01~ 전체 lot을 담아야 하지만 preprocessed는 RECENT_DAYS=20
# 윈도우라 고전 blk을 못 담음 → daily raw에서 (EQP,BLK) 블록 프로필을 별도 누적 캐시.
# 정확성: raw 파일(일별 ~24h 겹침/시간별 ~24h)이 블럭(최대 31.5h)보다 짧아 한 블럭이
#   여러 파일에 걸린다 → 전역 (EQP,BLK,TRACE_DTTS) dedup 후 블록당 dedup 행수(n)의
#   **n-max merge**(더 완전한 프로필 보존)로 부분 재계산이 완전한 것을 덮어쓰지 못하게 한다.
# 성능: 캐시 없으면 08-01 전체 raw 1회 스캔(첫 빌드 ~134초, worker 병렬). 이후 매 run은
#   최근 ALL_BLOCKS_REBUILD_WINDOW_DAYS(2)일 끝 파일만 스캔(수 초~분) + 오래된 블록 이월.
ALL_BLOCKS_FROM = pd.Timestamp('2026-08-01')  # 블록 시작 시각 기준 시계열 시작
ALL_BLOCKS_REBUILD_WINDOW_DAYS = 2            # 매 run 재계산 창 (일) - 최대 블럭 길이(31.5h)보다 넓게
ALL_BLOCKS_SKIP_THRESHOLD = 15                # 신규/변경 파일이 이 값 초과면 이번 run은 기존 캐시 유지
ALL_BLOCKS_CACHE = os.path.join(DATA_DIR, 'all_blocks_profile.parquet')
ALL_BLOCKS_MANIFEST = os.path.join(DATA_DIR, 'all_blocks_manifest.json')
ALL_BLOCKS_COLS = ['EQP_ID', 'BLK_NO', 'TRACE_DTTS', 'RECIPE',
                   'FRAME_IN_TEMP', 'SLURRY_IN_TEMP', 'SL-BOW-BF', 'SL-WARP-BF',
                   'SL-TTV-ALL', 'SL-THK-AV', 'SET_TENSION', 'WAITING_TIME', 'INGOT_LEN',
                   'SHIFT_AMOUNT_WIREGUIDE_L', 'SHIFT_AMOUNT_WIREGUIDE_R']
# bow/warp 평균 대상 행 필터 - preprocessed(dropna subset=numeric_cols)와 동일 조건 (MES 기준)
BOW_FILTER_COLS = ['FRAME_IN_TEMP', 'SLURRY_IN_TEMP', 'SHIFT_AMOUNT_WIREGUIDE_L',
                   'SHIFT_AMOUNT_WIREGUIDE_R', 'SL-BOW-BF', 'SL-WARP-BF']
# roll 조건 4원의 raw 원천 컬럼 (블럭 mean으로 캐시 - _roll_at과 동일하게 직전 10블럭 median)
ROLL_FDC_COLS = ['SET_TENSION', 'WAITING_TIME', 'INGOT_LEN']
PRED_CACHE = os.path.join(DATA_DIR, 'block_predictions.parquet')
OTHER_GROUP = 'Other'

# ── 대상 장비 필터: apn_eng_blocks.csv에 이 EQP_NM(31개 pilot 장비)만 포함 ──
# EQP_ID ≠ EQP_NM인 장비도 있음(TSWST03→TSWS03, TSWST09→TSWS19, ...) →
# eqp_이름.csv merge 후 **EQP_NM** 기준으로 필터. (사용자 요청 2026-10-06)
EQUIP_FILTER = {
    'BSWS28', 'BSWS34', 'BSWS35', 'BSWS45', 'BSWS48', 'BSWS51', 'BSWS54', 'BSWS56',
    'BSWS58', 'BSWS62', 'BSWS05', 'BSWS07', 'BSWS09', 'BSWS10', 'BSWS11', 'BSWS12',
    'BSWS13', 'BSWS27', 'BSWS33', 'BSWS59',
    'TSWS03', 'TSWS05', 'TSWS07', 'TSWS09', 'TSWS11', 'TSWS13', 'TSWS15', 'TSWS17',
    'TSWS19', 'TSWS21', 'TSWS23',
}

PCTS = [f'{v}pct' for v in (0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 99, 100)]
PCTS_A = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 99, 100]   # 숫자 포인트 (프로파일 보간용)

APN_GROUP = 'APN'
ENG_GROUP = "Eng'r 변경"
# setpoint 판정: 소수 1째자리(0.1°C) 반올림 비교
#   snapshot(시간별 저장 추천) : 12지점 전부 동일 (diff 0)
#   back(역산 재현)            : 역산 드리프트 1스텝(0.1) 허용
TOL_SNAP = 0.0
TOL_BACK = 0.1

if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass


# ──────────────────────────────────────────────
# setpoint 판정
# ──────────────────────────────────────────────
def _zone_match(row, rec_prefix, actual_prefix, tol):
    """12지점 소수 1째자리 비교. max 차이 ≤ tol 이면 match.
    row: dict - '{actual_prefix}_{p}' 와 'rec_{rec_prefix}_temp_{p}' 키."""
    n_diff, n_cmp = 0.0, 0
    for p in PCTS:
        act = row.get(f'{actual_prefix}_{p}')
        rec = row.get(f'rec_{rec_prefix}_temp_{p}')
        if act is None or rec is None or pd.isna(act) or pd.isna(rec):
            continue
        n_cmp += 1
        n_diff = max(n_diff, abs(round(float(act), 1) - round(float(rec), 1)))
    return (n_cmp > 0 and n_diff <= tol), n_diff, n_cmp


def _resolve_anchor(row, sub):
    """추천행 → anchor 블럭. (anchor, src) or (None, None)."""
    blk = row.get('latest_block_no')
    if blk is not None and not (isinstance(blk, float) and np.isnan(blk)) and blk in set(sub['BLK_NO']):
        return blk, 'block_no'
    wire = row.get('latest_wire')
    if wire is not None and not (isinstance(wire, float) and np.isnan(wire)):
        wb = sub[sub['NEW_WIRE_ID'] == wire]
        if len(wb):
            return wb['BLK_NO'].iloc[-1], 'wire'
    return None, None


# ──────────────────────────────────────────────
# 역산 (back) - 추천 저장 이전 lot 재현
# ──────────────────────────────────────────────
def _back_cfg():
    from recommend_future import RECOMMEND_CONFIG as C
    return C


_CFG = None


def _cfg():
    global _CFG
    if _CFG is None:
        _CFG = _back_cfg()
    return _CFG


def _pt_model_dir(pt_val):
    pt_dir = os.path.join(_cfg()['model_dir'], str(pt_val))
    return pt_dir if os.path.isdir(pt_dir) else _cfg()['model_dir']


def _roll_at(pq, eqp, pt_val, blk_time):
    """블럭 시작 시점 직전 N wire 중간값 roll 조건 (raw 컬럼명 매핑 포함).
    (recommend_future.compute_roll_for_latest_wire와 동일하게 중간값 - 센서 스파이크 방어)"""
    from recommend_future import RENAME_FOR_RECOMMEND
    sub = pq[(pq['EQP_ID'] == eqp) & (pq['process_time'] == pt_val)]
    prior = sub[sub['TRACE_DTTS'] < blk_time].sort_values('TRACE_DTTS').tail(_cfg()['roll_window'])
    if len(prior) == 0:
        return None
    prior = prior.rename(columns={v: k for k, v in RENAME_FOR_RECOMMEND.items()
                                  if k in prior.columns})
    if 'range_slurry_temp_10_0' not in prior.columns:
        low = {c.lower(): c for c in prior.columns}
        c10, c0 = low.get('slurry_in_temp_10pct'), low.get('slurry_in_temp_0pct')
        if c10 and c0:
            prior = prior.copy()
            prior['range_slurry_temp_10_0'] = prior[c10] - prior[c0]
    roll = {}
    for c in _cfg()['roll_source_cols']:
        if c in prior.columns:
            v = prior[c].median()
            if pd.notna(v):
                roll[f'roll_{c}'] = float(v)
    return roll or None


def _back_recommend(eqp, pt_val, roll, ref_frame=None):
    """시점 역산 추천 재현. 반환: (rec_dict, pred_f, pred_s) or None.
    rec_dict: {'rec_{zone}_temp_{p}': 2dp 세팅값}  pred: bow_with_recipe.
    ref_frame: 추천 시점의 최신 blk frame 프로파일 (12지점 °C) - ±0.5 clamp 재현용."""
    from recommend_future import _inverse_profile_with_predict, _clamp_frame_to_latest
    pt_dir = _pt_model_dir(pt_val)
    out = {}
    for name in ('frame', 'slurry'):
        inv, predict = _inverse_profile_with_predict(
            pt_dir, name, _cfg()['target_bow'], roll, eqp,
            frame_start=_cfg().get('frame_start_default'), cfg=_cfg())
        if inv is None:
            return None
        rec = inv['recipe']
        if name == 'frame':
            rec, bow, _shift, _dev = _clamp_frame_to_latest(
                rec, ref_frame, predict, _cfg()['target_bow'], _cfg())
        else:
            bow = inv['bow_with_recipe']
        for c, v in rec.items():
            if c.startswith('set_'):
                out[f'rec_{c[4:]}'] = v
        out[f'pred_{name}'] = bow
    return out, out['pred_frame'], out['pred_slurry']


def _ref_frame_before(pq, eqp, pt_val, blk_time):
    """blk_time 직전 blk의 frame 프로파일 (12지점 °C) - 추천 시점의 최신 blk 재현.
    pq가 blk당 1행이라 blk_time 미만 마지막 행이 바로 직전 blk."""
    sub = pq[(pq['EQP_ID'] == eqp) & (pq['process_time'] == pt_val)]
    prior = sub[sub['TRACE_DTTS'] < blk_time].sort_values('TRACE_DTTS')
    if len(prior) == 0:
        return None
    vals = []
    for p in PCTS:
        v = prior[f'FRAME_IN_TEMP_{p}'].iloc[-1]
        if v is None or pd.isna(v):
            return None
        vals.append(float(v))
    return vals


_FORWARD_CACHE = {}
def _forward(eqp, pt_val, name, roll, vals):
    """12지점 프로파일 → 모델 forward (예측 bow)."""
    from recommend_future import load_profile_model
    pt_dir = _pt_model_dir(pt_val)
    key = (pt_dir, name)
    if key not in _FORWARD_CACHE:
        _FORWARD_CACHE[key] = load_profile_model(pt_dir, name)
    L = _FORWARD_CACHE[key]
    if L is None:
        return None
    model, scaler, meta = L
    feats = meta['feature_cols']; xs = meta['x_stats']
    pc = meta['profile_cols']; rc = meta.get('roll_cols', [])
    ec = meta.get('eqp_cols', []); pf = meta.get('eqp_prefix', 'eqp_')
    oc = {c: i for i, c in enumerate(pc) if c in feats}

    def gv(c):
        if c in ec:
            return 1.0 if c == f'{pf}{eqp}' else 0.0
        if c in oc:
            return float(vals[oc[c]])
        if c in rc:
            v = roll.get(c)
            if v is not None and not pd.isna(v):
                return float(v)
            return float(xs.get(c, {}).get('mean', 0.0))
        return float(xs.get(c, {}).get('mean', 0.0))

    x = np.array([gv(c) for c in feats]).reshape(1, -1)
    return float(model.predict(scaler.transform(x))[0])


# ──────────────────────────────────────────────
# 컬럼 보강: EQP_NM / TTV / AVE_THK
# ──────────────────────────────────────────────
def load_eqp_nm():
    """eqp_이름.csv → [EQP_ID, EQP_NM] (eqp_id→EQP_ID, eqp_nm→EQP_NM)."""
    df = pd.read_csv(EQP_NM_CSV, encoding='utf-8')
    return df.rename(columns={'eqp_id': 'EQP_ID', 'eqp_nm': 'EQP_NM'})[['EQP_ID', 'EQP_NM']]


def build_ttv_thk_block_map(force=False):
    """by_eqp 폴더(장비별 raw)에서 (EQP_ID, BLK_NO) 단위 집계.
    TRACE_DTTS=min, TTV/AVE_THK/avg_bow/avg_warp = BLK_NO별 단순 mean() (dedup 없음, MES 기준).
    by_eqp에 없는 컬럼(예: split에 TTV/THK 미포함)은 NaN. force 인자는 호환용(미사용)."""
    src = {'TRACE_DTTS': 'TRACE_DTTS', 'SL-TTV-ALL': 'TTV', 'SL-THK-AV': 'AVE_THK',
           'SL-BOW-BF': 'avg_bow', 'SL-WARP-BF': 'avg_warp'}
    out_cols = ['EQP_ID', 'BLK_NO', 'TRACE_DTTS', 'TTV', 'AVE_THK', 'avg_bow', 'avg_warp']
    files = sorted(glob.glob(os.path.join(BY_EQP_DIR, '*.parquet')))
    parts = []
    for f in files:
        names = set(parq.ParquetFile(f).schema.names)
        use = ['EQP_ID', 'BLK_NO'] + [c for c in src if c in names]
        d = pd.read_parquet(f, columns=use)
        d = d.dropna(subset=['EQP_ID', 'BLK_NO'])
        if d.empty:
            continue
        for c in ('SL-TTV-ALL', 'SL-THK-AV', 'SL-BOW-BF', 'SL-WARP-BF'):
            if c in d.columns:
                d[c] = pd.to_numeric(d[c], errors='coerce')
        if 'TRACE_DTTS' in d.columns:
            d['TRACE_DTTS'] = pd.to_datetime(d['TRACE_DTTS'], errors='coerce')
        agg = {src[c]: (c, 'min' if c == 'TRACE_DTTS' else 'mean') for c in src if c in d.columns}
        g = d.groupby(['EQP_ID', 'BLK_NO']).agg(**agg).reset_index()
        parts.append(g.reindex(columns=out_cols))
    if not parts:
        print(f"  [BY_EQP] 장비 파일 없음: {BY_EQP_DIR}")
        return pd.DataFrame(columns=out_cols)
    out = pd.concat(parts, ignore_index=True).drop_duplicates(['EQP_ID', 'BLK_NO'])
    print(f"  [BY_EQP] 장비 {len(files)}개 → 블럭 {len(out)}개 (BLK_NO별 mean)")
    return out


# ──────────────────────────────────────────────
# 전체 히스토리 블록 프로필 ("Other" lot 보강용)
# ──────────────────────────────────────────────
def _all_blocks_file_date(path):
    """파일명 날짜 (per-day=8자리 / hourly=14자리). 못 읽으면 None."""
    m = re.search(r'_(20\d{6,14})\.parquet$', os.path.basename(path))
    if not m:
        return None
    try:
        return datetime.strptime(m.group(1), '%Y%m%d%H%M%S' if len(m.group(1)) == 14 else '%Y%m%d')
    except ValueError:
        return None


def _dedup_blocks_in_file(f):
    """하나의 daily raw 파일 → (EQP_ID,BLK_NO,TRACE_DTTS) dedup된 작은 프레임.
    raw는 (blk,ts)당 와이어별 중복 기록(~30x)이 있어 이 dedup으로 수백만 행 → ~20만 행."""
    d = parq.read_table(f, columns=ALL_BLOCKS_COLS).to_pandas()
    if d.empty:
        return pd.DataFrame(), pd.DataFrame()
    for c in ('FRAME_IN_TEMP', 'SLURRY_IN_TEMP', 'SL-BOW-BF', 'SL-WARP-BF',
              'SL-TTV-ALL', 'SL-THK-AV', 'SHIFT_AMOUNT_WIREGUIDE_L',
              'SHIFT_AMOUNT_WIREGUIDE_R') + tuple(ROLL_FDC_COLS):
        d[c] = pd.to_numeric(d[c], errors='coerce')
    d = d.dropna(subset=['EQP_ID', 'BLK_NO', 'TRACE_DTTS'])
    if d.empty:
        return pd.DataFrame(), pd.DataFrame()
    # bow/warp는 dedup 전(웨이퍼 fan-out 유지) 합계/행수 - preprocessed·MES 기준과 동일
    v = d.dropna(subset=BOW_FILTER_COLS)
    agg = (v.groupby(['EQP_ID', 'BLK_NO'])
           .agg(bow_sum=('SL-BOW-BF', 'sum'), warp_sum=('SL-WARP-BF', 'sum'),
                n_raw=('SL-BOW-BF', 'size'))
           .reset_index())
    d = (d.sort_values(['EQP_ID', 'BLK_NO', 'TRACE_DTTS'], kind='stable')
         .drop_duplicates(['EQP_ID', 'BLK_NO', 'TRACE_DTTS'], keep='first'))
    return d.reset_index(drop=True), agg


def _block_profile_series(grp):
    """(EQP,BLK) 그룹(시간순) → 1행 프로필 Series. 12지점은 인덱스 보간(np.interp).
    n=블록 dedup 행수: 같은 블록의 재계산 중 더 완전한(큰 n) 프로필을 보존하는 용도."""
    row = {'n': len(grp), 'TRACE_DTTS': grp['TRACE_DTTS'].min()}
    rc = grp['RECIPE'].dropna()
    rcs = str(rc.iloc[0]) if len(rc) else ''
    if any(k in rcs for k in ('133', '150', '151')):
        row['process_time'] = '13.3Hr'
    elif any(k in rcs for k in ('180', '181', '185')):
        row['process_time'] = '18.5Hr'
    elif rcs:
        row['process_time'] = 'etc'
    else:
        row['process_time'] = None
    for col, key in (('SL-BOW-BF', 'avg_bow'), ('SL-WARP-BF', 'avg_warp'),
                     ('SL-TTV-ALL', 'TTV'), ('SL-THK-AV', 'AVE_THK')):
        row[key] = grp[col].mean()
    # roll 조건 원천 (블럭 mean - _roll_at과 동일하게 직전 N블럭 median 재구성용)
    for col, key in (('SET_TENSION', 'fdc_set_tension'), ('INGOT_LEN', 'fdc_ingot_len'),
                     ('WAITING_TIME', 'fdc_wait_time')):
        if col in grp.columns:
            row[key] = grp[col].mean()
    for col, pre in (('FRAME_IN_TEMP', 'act_frame'), ('SLURRY_IN_TEMP', 'act_slurry')):
        vals = grp[col].dropna().to_numpy()
        for p in PCTS_A:
            if len(vals) == 0:
                v = np.nan
            elif len(vals) == 1:
                v = float(vals[0])
            else:
                v = float(np.interp(p / 100.0, np.linspace(0.0, 1.0, len(vals)), vals))
            row[f'{pre}_{p}pct'] = round(v, 3)
    return pd.Series(row)


def build_all_blocks_profile(force=False):
    """daily raw 전체(08-01~)에서 (EQP_ID, BLK_NO) 블록 프로필 → 누적 캐시.
    preprocessed는 RECENT_DAYS=20 윈도우라 08-01~ 전체를 담지 못해서 "Other" lot에도
    실측 12지점 온도·bow·warp·TTV·THK를 채워야 하는 용도.

    정확성: raw 파일(시간별 ~24h 커버)이 블럭(최대 31.5h)보다 짧아 한 블럭이 여러 파일에
    걸린다. hence **창 union + 전역 dedup**으로 합치고, 블록당 dedup 행수(n)로
    **n-max merge**(더 완전한 프로필 보존)로 부분 재계산이 완전한 것을 덮어쓰지 못하게 한다.
    매시간 실행 시 블럭 완료 후 ~40시간 내 완전 프로파일로 자가 수정된다.

    성능 (사용자 확정 2026-10-06 누적 캐시):
      - 캐시 없으면 08-01 전체 raw 1회 스캔(첫 빌드, 수 분, worker 16). 이후 매 run은
        최근 ALL_BLOCKS_REBUILD_WINDOW_DAYS(2)일 끝 파일만 스캔(수 초~분).
      - 신규/변경 파일이 ALL_BLOCKS_SKIP_THRESHOLD 초과면 이번 run은 기존 캐시 유지(다음 run 반영).
    반환: DataFrame(EQP_ID,BLK_NO,TRACE_DTTS,process_time,avg_bow,avg_warp,TTV,AVE_THK,
                    act_frame_{p}pct, act_slurry_{p}pct)."""
    files = sorted(glob.glob(DAILY_GLOB))
    files = [f for f in files if (d := _all_blocks_file_date(f)) is None or d >= ALL_BLOCKS_FROM]
    sig = {os.path.basename(f): os.path.getsize(f) for f in files}
    os.makedirs(DATA_DIR, exist_ok=True)
    man = {}
    if os.path.exists(ALL_BLOCKS_MANIFEST):
        with open(ALL_BLOCKS_MANIFEST, encoding='utf-8') as fh:
            man = json.load(fh)
    done = man.get('sig', {})
    # 스키마 버전 (v2: FDC 3컬럼 추가 - roll 재구성용)
    cache = (pd.read_parquet(ALL_BLOCKS_CACHE)
             if done and man.get('v') == 3 and os.path.exists(ALL_BLOCKS_CACHE) else pd.DataFrame())

    new_files = [f for f in files if done.get(os.path.basename(f)) != sig[os.path.basename(f)]]
    if not new_files and len(cache):
        print(f"  [ALL-BLK] 캐시 유효 (파일 {len(done)}개, 블록 {len(cache)}개)")
        return cache
    if len(cache) and len(new_files) > ALL_BLOCKS_SKIP_THRESHOLD:
        print(f"  [ALL-BLK] 변경 파일 {len(new_files)}개 - 이번 run은 기존 캐시({len(cache)}블록) 유지, 다음 run 반영")
        return cache

    t0 = time.time()
    if len(cache):
        # 증분: 최근 2일 끝 파일 union + 신규/변경 파일
        from datetime import timedelta
        win_end = datetime.now() - timedelta(days=ALL_BLOCKS_REBUILD_WINDOW_DAYS)
        rebuild = [f for f in files if (_all_blocks_file_date(f) or datetime.min) >= win_end]
        rebuild += [f for f in new_files if f not in rebuild]
        mode = f"증분 {len(rebuild)}파일 (최근 {ALL_BLOCKS_REBUILD_WINDOW_DAYS}일~)"
    else:
        rebuild = files
        mode = f"첫 전체 {len(rebuild)}파일 (08-01~)"
    print(f"  [ALL-BLK] {mode} - dedup·12지점 보간 (worker {min(8, len(rebuild))}) ...")

    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=min(12, len(rebuild))) as ex:
        results = list(ex.map(_dedup_blocks_in_file, rebuild))
    parts = [p for p, _ in results if p is not None and len(p)]
    aggs = [a for _, a in results if a is not None and len(a)]
    union = pd.concat(parts, ignore_index=True)
    # 전역 dedup: 인접 파일 겹침(블럭이 2파일로 분리) 해결 - (EQP,BLK,TRACE_DTTS) 1행만
    union = (union.sort_values(['EQP_ID', 'BLK_NO', 'TRACE_DTTS'], kind='stable')
             .drop_duplicates(['EQP_ID', 'BLK_NO', 'TRACE_DTTS'], keep='first'))
    recent = (union.groupby(['EQP_ID', 'BLK_NO'], sort=False)
              .apply(_block_profile_series, include_groups=False)
              .reset_index())
    # bow/warp를 MES 기준(dedup 전 전체 행 평균)으로 교체
    if aggs:
        ag = (pd.concat(aggs, ignore_index=True)
              .groupby(['EQP_ID', 'BLK_NO'], as_index=False)[['bow_sum', 'warp_sum', 'n_raw']].sum())
        ag['avg_bow_mes'] = ag['bow_sum'] / ag['n_raw']
        ag['avg_warp_mes'] = ag['warp_sum'] / ag['n_raw']
        recent = recent.merge(ag[['EQP_ID', 'BLK_NO', 'avg_bow_mes', 'avg_warp_mes']],
                              on=['EQP_ID', 'BLK_NO'], how='left')
        recent['avg_bow'] = recent['avg_bow_mes']
        recent['avg_warp'] = recent['avg_warp_mes']
        recent = recent.drop(columns=['avg_bow_mes', 'avg_warp_mes'])
    if len(cache):
        # n-max merge: 같은 블록의 재계산이 더 완전할 때만 대체 (부분 재계산이
        # 완전한 캐시를 덮어쓰지 못하게). 블록 진행 중 n은 항상 증가 → 자기수정.
        def _merge(agg):
            return agg.loc[agg['n'].idxmax()]
        cache = pd.concat([cache, recent], ignore_index=True) \
                  .groupby(['EQP_ID', 'BLK_NO'], sort=False).apply(_merge, include_groups=False) \
                  .reset_index()
    else:
        cache = recent
    # 시계열 시작(08-01) 이전 블록 제외 (0801 파일이 07-30~부터 담김)
    cache = cache[cache['TRACE_DTTS'] >= ALL_BLOCKS_FROM]
    cache = cache.reset_index(drop=True)
    cache.to_parquet(ALL_BLOCKS_CACHE, index=False)
    with open(ALL_BLOCKS_MANIFEST, 'w', encoding='utf-8') as fh:
        json.dump({'sig': dict(sig), 'v': 3}, fh)
    print(f"  [ALL-BLK] union dedup {len(union):,}행 → 블록 {len(cache)}개 ({time.time()-t0:.0f}s)")
    return cache


# ──────────────────────────────────────────────
# 모델 예측 보강 (pred_rec/pred_actual 누락 블록)
# ──────────────────────────────────────────────
PRED_REBUILD_WINDOW_DAYS = 3   # 예측 재계산 창 - 확정된 오래된 블록은 캐시 유지

def _model_eqps_by_pt():
    """process_time → 모델(eqp 더미)에 들어 있는 장비 set. 모델 없는 pt는 빈 set."""
    out = {}
    for pt in _cfg()['process_times']:
        mp = os.path.join(_pt_model_dir(pt), 'frame', 'meta.json')
        if not os.path.exists(mp):
            out[pt] = set()
            continue
        with open(mp, encoding='utf-8') as fh:
            meta = json.load(fh)
        pfx = meta.get('eqp_prefix', 'eqp_')
        out[pt] = {c[len(pfx):] for c in meta.get('eqp_cols', [])}
    return out


def _predict_group(group_df):
    """(EQP_ID, process_time) 그룹(TRACE_DTTS 오름차순) → 블록당 pred 1행.
    roll = 직전 10블럭 FDC 4원 median (_roll_at 동일), ref_frame = 직전 blk frame 12지점
    (±0.5 clamp 재현, 첫 blk은 None). 역산(_back_recommend)+실측 forward(_forward).
    (EQP_ID, process_time) 그룹 단위라 _roll_at의 process_time 필터와 동일 -
    그룹 첫 blk(직전 blk 없음)은 pred NaN (원 back 경로도 '직전 wire 없음'으로 제외)."""
    eqp = group_df['EQP_ID'].iloc[0]
    pt_val = group_df['process_time'].iloc[0]
    rw = _cfg()['roll_window']
    rows = []
    for i in range(len(group_df)):
        b = group_df.iloc[i]
        if i == 0:
            rows.append({'EQP_ID': eqp, 'BLK_NO': b['BLK_NO'], 'pred_rec': np.nan,
                         'pred_actual': np.nan})
            continue
        prior = group_df.iloc[max(0, i - rw):i]
        roll = {}
        for c, key in (('fdc_set_tension', 'roll_fdc_set_tension'),
                       ('fdc_wait_time', 'roll_fdc_wait_time'),
                       ('fdc_ingot_len', 'roll_fdc_ingot_len')):
            if c in prior.columns:
                v = prior[c].median()
                if pd.notna(v):
                    roll[key] = float(v)
        # range_slurry_temp_10_0: _roll_at과 동일하게 행당 차이를 먼저 구한 뒤 median
        rng = prior['act_slurry_10pct'] - prior['act_slurry_0pct']
        if rng.notna().any():
            roll['roll_range_slurry_temp_10_0'] = float(rng.median())
        pb = group_df.iloc[i - 1]
        ref = [pb[f'act_frame_{p}'] for p in PCTS]
        if any(pd.isna(v) for v in ref):
            ref = None
        try:
            bk = _back_recommend(eqp, pt_val, roll, ref_frame=ref)
        except Exception:
            bk = None
        if bk is None:
            rows.append({'EQP_ID': eqp, 'BLK_NO': b['BLK_NO'], 'pred_rec': np.nan,
                         'pred_actual': np.nan})
            continue
        _rec, pf, ps = bk
        pred_rec = round((pf + ps) / 2, 3)
        fv = [b[f'act_frame_{p}'] for p in PCTS]
        sv = [b[f'act_slurry_{p}'] for p in PCTS]
        pfa = psa = None
        if not any(pd.isna(v) for v in fv):
            pfa = _forward(eqp, pt_val, 'frame', roll, fv)
        if not any(pd.isna(v) for v in sv):
            psa = _forward(eqp, pt_val, 'slurry', roll, sv)
        pred_actual = round((pfa + psa) / 2, 3) if (pfa is not None and psa is not None) else np.nan
        rows.append({'EQP_ID': eqp, 'BLK_NO': b['BLK_NO'], 'pred_rec': pred_rec,
                     'pred_actual': pred_actual})
    return pd.DataFrame(rows)


def _predict_group_job(job):
    """(df, n_ctx) → 컨텍스트(n_ctx 행)를 제외한 블록 pred (spawn pickling용 모듈 함수)."""
    df, n_ctx = job
    return _predict_group(df).iloc[n_ctx:]


def build_predictions(prof):
    """feasible 블록(모델 있는 BSWS 13.3/18.5) 전체의 pred_rec/pred_actual → 캐시.
    첫 run은 feasible 전부(수 분, worker 12 병렬). 이후 매 run은 최근
    PRED_REBUILD_WINDOW_DAYS(3)일 블록만 재계산(roll 조건이 직전 blk 완료로 바뀌므로),
    확정된 오래된 블록은 캐시 유지.
    반환: DataFrame(EQP_ID, BLK_NO, pred_rec, pred_actual)."""
    if len(prof) == 0:
        return pd.DataFrame(columns=['EQP_ID', 'BLK_NO', 'pred_rec', 'pred_actual'])
    model_eqps = _model_eqps_by_pt()
    feas = prof[prof.apply(lambda r: r['process_time'] in model_eqps
                           and r['EQP_ID'] in model_eqps.get(r['process_time'], set()),
                           axis=1)]
    if len(feas) == 0:
        return pd.DataFrame(columns=['EQP_ID', 'BLK_NO', 'pred_rec', 'pred_actual'])
    pred = pd.read_parquet(PRED_CACHE) if os.path.exists(PRED_CACHE) else pd.DataFrame()
    from datetime import timedelta
    win_start = pd.Timestamp(datetime.now() - timedelta(days=PRED_REBUILD_WINDOW_DAYS))
    if len(pred):
        # 값이 든 캐시 행만 제외; NaN 행(진행 중·실패)은 재시도
        ok_idx = pred[pred['pred_rec'].notna()].set_index(['EQP_ID', 'BLK_NO']).index
        todo = feas[(feas['TRACE_DTTS'] >= win_start) &
                    ~feas.set_index(['EQP_ID', 'BLK_NO']).index.isin(ok_idx)]
    else:
        todo = feas   # 첫 run: feasible 전부
    if len(todo):
        t0 = time.time()
        print(f"  [PRED] 블록 {len(todo)}개 예측 (그룹 {todo[['EQP_ID', 'process_time']].drop_duplicates().shape[0]}개) ...")
        todo = todo.sort_values(['EQP_ID', 'process_time', 'TRACE_DTTS'])
        rw = _cfg()['roll_window']
        # 직전 roll_window 블록을 컨텍스트로 포함 → 증분 run에서 그룹 첫 블록도
        # 직전 blk(캐시에 확정)이 있으면 roll·ref_frame으로 정상 예측
        groups = []
        for _, grp in todo.groupby(['EQP_ID', 'process_time'], sort=False):
            eqp, ptv, first_ts = grp['EQP_ID'].iloc[0], grp['process_time'].iloc[0], grp['TRACE_DTTS'].iloc[0]
            ctx = feas[(feas['EQP_ID'] == eqp) & (feas['process_time'] == ptv)
                       & (feas['TRACE_DTTS'] < first_ts)].tail(rw)
            groups.append((pd.concat([ctx, grp], ignore_index=True) if len(ctx) else grp, len(ctx)))
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=12) as ex:
            parts = list(ex.map(_predict_group_job, groups))
        new = pd.concat(parts, ignore_index=True)
        pred = (pd.concat([pred, new], ignore_index=True)
                .drop_duplicates(['EQP_ID', 'BLK_NO'], keep='last') if len(pred) else new)
        os.makedirs(DATA_DIR, exist_ok=True)
        pred.to_parquet(PRED_CACHE, index=False)
        ok = int(pred['pred_rec'].notna().sum())
        print(f"  [PRED] {time.time()-t0:.0f}s - 누적 {len(pred)}개 (예측 성공 {ok})")
    else:
        print(f"  [PRED] 재계산 블록 없음 (캐시 {len(pred)}개 유지)")
    return pred

# ──────────────────────────────────────────────
# 1) 시간별 규칙: anchor 다음다음 run → APN_LOT.csv append
# ──────────────────────────────────────────────
def hourly_classify(pq, apn_existing):
    """data/recommend 추천 파일 전체 → anchor(idx)+2 블럭 판정.
    기존 APN_LOT 에 있는 lot 은 스킵(수동 우선).
    반환: [(LOT_ID, EQP_ID, group), ...] 신규 행."""
    existing = set(zip(apn_existing['LOT_ID'], apn_existing['EQP_ID'])) if len(apn_existing) else set()
    files = sorted(glob.glob(os.path.join(REC_DIR, 'recommend_future_*.csv')))
    new_rows, seen = [], set()
    for f in files:
        df = pd.read_csv(f, encoding='utf-8-sig')
        fname = os.path.basename(f)
        n_hit = 0
        for _, r in df.iterrows():
            eqp, pt_val = r['eqp'], r['process_time']
            sub = pq[(pq['EQP_ID'] == eqp) & (pq['process_time'] == pt_val)]
            if len(sub) < 3:
                continue
            sub = sub.sort_values('TRACE_DTTS').reset_index(drop=True)
            anchor, src = _resolve_anchor(r, sub)
            if anchor is None:
                continue
            idx = int(sub.index[sub['BLK_NO'] == anchor][0])
            # 다음다음 run = anchor + 2
            if idx + 2 >= len(sub):
                continue   # 아직 안 돌았음
            tgt = sub.iloc[idx + 2]
            key = (tgt['BLK_NO'], eqp)
            if key in existing or key in seen:
                continue
            merged = tgt.to_dict()
            for p in PCTS:
                merged[f'rec_frame_temp_{p}'] = r.get(f'rec_set_frame_temp_{p}')
                merged[f'rec_slurry_temp_{p}'] = r.get(f'rec_set_slurry_temp_{p}')
            fm, fd, fn = _zone_match(merged, 'frame', 'FRAME_IN_TEMP', TOL_SNAP)
            sm, sd, sn = _zone_match(merged, 'slurry', 'SLURRY_IN_TEMP', TOL_SNAP)
            group = APN_GROUP if (fm or sm) else ENG_GROUP
            new_rows.append({'LOT_ID': tgt['BLK_NO'], 'Tuning_Group': group, 'EQP_ID': eqp})
            seen.add(key)
            n_hit += 1
            print(f"    {fname} {eqp} [{pt_val}]: anchor={anchor}({src}) +2 = {tgt['BLK_NO']} "
                  f"({tgt['TRACE_DTTS']:%m-%d %H:%M}) → {group} (frameΔ{fd:.1f}/slurryΔ{sd:.1f})")
        if n_hit:
            print(f"  {fname}: {n_hit}개 신규 판정")
    return pd.DataFrame(new_rows, columns=['LOT_ID', 'Tuning_Group', 'EQP_ID']) if new_rows else pd.DataFrame()

def _append_apn_lot(new, path):
    """APN_LOT.csv append - (LOT_ID,EQP_ID) 중복 시 기존 행 유지(수동 우선)."""
    if os.path.exists(path):
        old = pd.read_csv(path, encoding='utf-8-sig')
    else:
        old = pd.DataFrame(columns=['LOT_ID', 'Tuning_Group', 'EQP_ID'])
    if len(new) == 0:
        old.to_csv(path, index=False, encoding='utf-8-sig')
        return len(old), 0
    merged = pd.concat([old, new], ignore_index=True)
    before = len(merged)
    merged = merged.drop_duplicates(subset=['LOT_ID', 'EQP_ID'], keep='first').reset_index(drop=True)
    merged.to_csv(path, index=False, encoding='utf-8-sig')
    return len(merged), before - len(merged)

# ──────────────────────────────────────────────
# 2) 리포트 데이터: APN_LOT lot → 실측·예측·오차
# ──────────────────────────────────────────────
def _pilot_zone_diff(row, name):
    """pilot_results 행 → (rec, act) 12지점 소수 1째자리 max 차이."""
    ds = []
    for p in PCTS:
        rr, aa = row.get(f'rec_{name}_{p}'), row.get(f'act_{name}_{p}')
        if rr is None or aa is None or pd.isna(rr) or pd.isna(aa):
            continue
        ds.append(abs(round(float(aa), 1) - round(float(rr), 1)))
    return max(ds) if ds else None


def build_report_data(apn_lot, pq):
    """APN_LOT 의 lot마다 판정 상세 행 생성 (pilot join → parquet back 순)."""
    pilot = pd.read_csv(PILOT_RESULTS, encoding='utf-8-sig') if os.path.exists(PILOT_RESULTS) else pd.DataFrame()
    pmap = {r['blk']: r for _, r in pilot.iterrows()} if len(pilot) else {}

    rows = []
    for _, lot in apn_lot.iterrows():
        blk, eqp = lot['LOT_ID'], lot['EQP_ID']
        grp = lot['Tuning_Group']

        if blk in pmap:
            pr = pmap[blk]
            pt_val = pr['pt']
            fd = _pilot_zone_diff(pr, 'frame')
            sd = _pilot_zone_diff(pr, 'slurry')
            fm = fd is not None and fd <= TOL_BACK
            sm = sd is not None and sd <= TOL_BACK   # pilot 역산도 back과 같은 재현
            row = {
                'LOT_ID': blk, 'EQP_ID': eqp, 'process_time': pt_val,
                'Tuning_Group': grp, 'group_src': 'csv',
                'source': 'pilot',
                'frame_match': bool(fm), 'slurry_match': bool(sm),
                'frame_diff_max': round(fd, 1) if fd is not None else None,
                'slurry_diff_max': round(sd, 1) if sd is not None else None,
                'actual_bow': round(float(pr['actual_bow']), 3),
                'actual_warp': round(float(pr['actual_warp']), 3) if pd.notna(pr['actual_warp']) else None,
                'pred_rec': round(float(pr['pred_rec']), 3),
                'pred_actual': round(float(pr['pred_actual']), 3),
                'err_rec': round(float(pr['err_rec']), 3),
                'err_actual': round(float(pr['err_actual']), 3),
            }
            # ① 레시피 프로파일 (12지점, 실측)
            for name in ('frame', 'slurry'):
                for p in PCTS:
                    v = pr.get(f'act_{name}_{p}')
                    row[f'act_{name}_{p}'] = round(float(v), 3) if v is not None and not pd.isna(v) else None
            rows.append(row)
            continue

        # pilot에 없으면 parquet 역산
        sub = pq[(pq['EQP_ID'] == eqp) & (pq['BLK_NO'] == blk)]
        if len(sub) == 0:
            print(f"    {blk} {eqp}: pilot/parquet 둘 다 없음 - 리포트 제외")
            continue
        b = sub.sort_values('TRACE_DTTS').iloc[0]
        pt_val = b['process_time']
        roll = _roll_at(pq, eqp, pt_val, b['TRACE_DTTS'])
        if roll is None:
            print(f"    {blk} {eqp}: 직전 wire 없음 - 역산 불가, 리포트 제외")
            continue
        ref_frame = _ref_frame_before(pq, eqp, pt_val, b['TRACE_DTTS'])
        bk = _back_recommend(eqp, pt_val, roll, ref_frame=ref_frame)
        if bk is None:
            print(f"    {blk} {eqp}: 역산 실패 - 리포트 제외")
            continue
        rec_dict, pred_f, pred_s = bk
        merged = b.to_dict()
        merged.update(rec_dict)
        fm, fd, fn = _zone_match(merged, 'frame', 'FRAME_IN_TEMP', TOL_BACK)
        sm, sd, sn = _zone_match(merged, 'slurry', 'SLURRY_IN_TEMP', TOL_BACK)
        ab = float(b['avg_bow_bf_total']) if pd.notna(b['avg_bow_bf_total']) else np.nan
        # pred_actual: 실측 레시피 forward
        pfa, psa = None, None
        fv = [b.get(f'FRAME_IN_TEMP_{p}') for p in PCTS]
        sv = [b.get(f'SLURRY_IN_TEMP_{p}') for p in PCTS]
        if all(v is not None and not pd.isna(v) for v in fv):
            pfa = _forward(eqp, pt_val, 'frame', roll, fv)
        if all(v is not None and not pd.isna(v) for v in sv):
            psa = _forward(eqp, pt_val, 'slurry', roll, sv)
        pred_rec = (pred_f + pred_s) / 2
        pred_actual = (pfa + psa) / 2 if (pfa is not None and psa is not None) else np.nan
        aw = b.get('avg_warp_bf_total')
        row = {
            'LOT_ID': blk, 'EQP_ID': eqp, 'process_time': pt_val,
            'Tuning_Group': grp, 'group_src': 'csv',
            'source': 'back',
            'frame_match': bool(fm), 'slurry_match': bool(sm),
            'frame_diff_max': round(fd, 1) if fn else None,
            'slurry_diff_max': round(sd, 1) if sn else None,
            'actual_bow': round(ab, 3) if pd.notna(ab) else None,
            'actual_warp': round(float(aw), 3) if aw is not None and pd.notna(aw) else None,
            'pred_rec': round(float(pred_rec), 3),
            'pred_actual': round(float(pred_actual), 3) if pd.notna(pred_actual) else None,
            'err_rec': round(ab - pred_rec, 3) if pd.notna(ab) else None,
            'err_actual': round(ab - pred_actual, 3) if (pd.notna(ab) and pd.notna(pred_actual)) else None,
        }
        for name in ('frame', 'slurry'):
            for p in PCTS:
                v = b.get(f'{name.upper()}_IN_TEMP_{p}')
                row[f'act_{name}_{p}'] = round(float(v), 3) if v is not None and not pd.isna(v) else None
        rows.append(row)
    return pd.DataFrame(rows)

def _apply_pq_values(o, pq, bym=None):
    """2b(APN_LOT 밖) 행 중 preprocessed(pq)에 있는 블록은 pq 값으로 교체.
    actual_bow/actual_warp = avg_bow_bf_total/avg_warp_bf_total (MES 기준),
    act_frame/act_slurry 12지점 = FRAME_IN_TEMP_{p}/SLURRY_IN_TEMP_{p}, process_time도 pq 값.
    pq에 없는 블록만 daily raw(prof) 값 유지. data_src 컬럼으로 출처 표시."""
    cols = ['EQP_ID', 'BLK_NO', 'process_time', 'avg_bow_bf_total', 'avg_warp_bf_total'] + \
           [f'FRAME_IN_TEMP_{p}' for p in PCTS] + [f'SLURRY_IN_TEMP_{p}' for p in PCTS]
    cols = [c for c in cols if c in pq.columns]
    v = pq[cols].drop_duplicates(['EQP_ID', 'BLK_NO']).rename(columns={'BLK_NO': 'LOT_ID'})
    pairs = [('avg_bow_bf_total', 'actual_bow'), ('avg_warp_bf_total', 'actual_warp'),
             ('process_time', 'process_time')]
    pairs += [(f'FRAME_IN_TEMP_{p}', f'act_frame_{p}') for p in PCTS]
    pairs += [(f'SLURRY_IN_TEMP_{p}', f'act_slurry_{p}') for p in PCTS]
    pairs = [(a, b) for a, b in pairs if a in v.columns]
    v = v.rename(columns={a: f'__pq_{a}' for a, _ in pairs})
    o = o.merge(v, on=['EQP_ID', 'LOT_ID'], how='left', indicator='__hit')
    hit = (o['__hit'] == 'both').to_numpy()
    for a, b in pairs:
        if b not in o.columns:
            o[b] = np.nan
        o[b] = o[b].astype(object)
        o.loc[hit, b] = o.loc[hit, f'__pq_{a}']
    o['data_src'] = np.where(hit, 'preprocessed', 'raw')
    o = o.drop(columns=[f'__pq_{a}' for a, _ in pairs] + ['__hit'])
    # pq에 없는 블록의 bow/warp, 그리고 전 블록의 TTV/AVE_THK는 by_eqp BLK_NO별 mean
    if bym is not None and len(bym):
        b = (bym[['EQP_ID', 'BLK_NO', 'avg_bow', 'avg_warp', 'TTV', 'AVE_THK']]
             .rename(columns={'BLK_NO': 'LOT_ID', 'avg_bow': '__b_bow', 'avg_warp': '__b_warp',
                              'TTV': '__b_ttv', 'AVE_THK': '__b_thk'}))
        o = o.merge(b, on=['EQP_ID', 'LOT_ID'], how='left', indicator='__bh')
        bh = (o['__bh'] == 'both').to_numpy()
        use = bh & (o['data_src'] == 'raw').to_numpy()
        for c in ('actual_bow', 'actual_warp', 'TTV', 'AVE_THK'):
            if c not in o.columns:
                o[c] = np.nan
        o.loc[use, 'actual_bow'] = o.loc[use, '__b_bow']
        o.loc[use, 'actual_warp'] = o.loc[use, '__b_warp']
        o.loc[bh, 'TTV'] = o.loc[bh, '__b_ttv']
        o.loc[bh, 'AVE_THK'] = o.loc[bh, '__b_thk']
        o.loc[use, 'data_src'] = 'by_eqp'
        o = o.drop(columns=['__b_bow', '__b_warp', '__b_ttv', '__b_thk', '__bh'])
    return o


def _load_existing_blocks(path):
    """기존 apn_eng_blocks.csv 로드 (없으면 빈 DataFrame)."""
    if not os.path.exists(path):
        return pd.DataFrame()
    old = pd.read_csv(path, encoding='utf-8-sig')
    if 'TRACE_DTTS' in old.columns:
        old['TRACE_DTTS'] = pd.to_datetime(old['TRACE_DTTS'], errors='coerce')
    return old


def _merge_with_existing(new, old, pq):
    """preprocessed(pq)에 있는 블록만 갱신/추가하고, 그 외 기존 행은 그대로 유지.
    pq에 있더라도 이번 run에서 재계산되지 못한 블록(역산 실패 등)은 기존 행 유지."""
    new = new.drop_duplicates(['EQP_ID', 'LOT_ID'], keep='last')
    if len(old) == 0:
        return new, len(new), 0, 0
    pq_keys = set(zip(pq['EQP_ID'], pq['BLK_NO']))
    new_in = new[[k in pq_keys for k in zip(new['EQP_ID'], new['LOT_ID'])]]
    new_keys = set(zip(new_in['EQP_ID'], new_in['LOT_ID']))
    old_keys = set(zip(old['EQP_ID'], old['LOT_ID']))
    keep_old = old[[k not in new_keys for k in zip(old['EQP_ID'], old['LOT_ID'])]]
    n_upd = len(new_keys & old_keys)
    n_add = len(new_keys) - n_upd
    merged = pd.concat([keep_old, new_in], ignore_index=True)
    return merged, n_add, n_upd, len(keep_old)


def main():
    pq = pd.read_parquet(PQ_PATH)
    pq['TRACE_DTTS'] = pd.to_datetime(pq['TRACE_DTTS'])
    print(f'실측: {PQ_PATH} ({pq["TRACE_DTTS"].min()} ~ {pq["TRACE_DTTS"].max()})')

    # 1) 시간별 규칙 → APN_LOT.csv append
    apn_old = pd.read_csv(APN_LOT_CSV, encoding='utf-8-sig') if os.path.exists(APN_LOT_CSV) else pd.DataFrame()
    new_rows = hourly_classify(pq, apn_old)
    n_total, n_new = _append_apn_lot(new_rows, APN_LOT_CSV)
    print(f"\n [1] 시간별 규칙: 신규 {n_new}개 판정 → APN_LOT.csv (총 {n_total}행)")
    if n_new == 0:
        print('      신규 없음 (대상 블럭 아직 실측에 없거나 전부 기존 lot)')

    # 2) 리포트 데이터 재생성
    apn_lot = pd.read_csv(APN_LOT_CSV, encoding='utf-8-sig')
    rep = build_report_data(apn_lot, pq)

    # ── 컬럼 보강: EQP_NM / TRACE_DTTS / TTV / AVE_THK ──
    # TRACE_DTTS: preprocessed(pq)에 블럭당 1행으로 존재 (blk 시작 시각)
    # TTV/AVE_THK: daily raw에서 (EQP_ID,BLK_NO) mean 집계 (캐시)
    # EQP_NM: eqp_이름.csv EQP_ID merge
    # ※ pq는 concat RECENT_DAYS=20 윈도우라 고전 lot은 아예 행이 없음 →
    #   그 lot은 daily raw(ttvm)에서 TRACE(min)/TTV/AVE_THK로 백필
    ttvm = build_ttv_thk_block_map()
    block_info = pq[['EQP_ID', 'BLK_NO', 'TRACE_DTTS']].copy()
    block_info = block_info.merge(ttvm[['EQP_ID', 'BLK_NO', 'TTV', 'AVE_THK']],
                                  on=['EQP_ID', 'BLK_NO'], how='left')
    block_info = block_info.drop_duplicates(['EQP_ID', 'BLK_NO'])
    # pq에 없는 lot은 daily raw 캐시(prof)에서 백필
    prof = build_all_blocks_profile()
    if len(prof) and 'TRACE_DTTS' in prof.columns:
        prof['TRACE_DTTS'] = pd.to_datetime(prof['TRACE_DTTS'])
    rep_keys = rep[['EQP_ID', 'LOT_ID']].rename(columns={'LOT_ID': 'BLK_NO'}).drop_duplicates()
    have = block_info[['EQP_ID', 'BLK_NO']]
    miss = rep_keys.merge(have, on=['EQP_ID', 'BLK_NO'], how='left', indicator=True)
    miss = miss[miss['_merge'] == 'left_only'].drop(columns=['_merge'])
    if len(miss) and (len(ttvm) or len(prof)):
        # by_eqp(BLK_NO별 mean) 우선, 없으면 daily raw 캐시(prof)
        cand = pd.concat([ttvm[['EQP_ID', 'BLK_NO', 'TRACE_DTTS', 'TTV', 'AVE_THK']],
                          prof[['EQP_ID', 'BLK_NO', 'TRACE_DTTS', 'TTV', 'AVE_THK']] if len(prof)
                          else pd.DataFrame()], ignore_index=True).drop_duplicates(['EQP_ID', 'BLK_NO'])
        bf = miss.merge(cand, on=['EQP_ID', 'BLK_NO'], how='left')
        block_info = pd.concat([block_info, bf], ignore_index=True)
        block_info = block_info.drop_duplicates(['EQP_ID', 'BLK_NO'])
        print(f"   [백필] preprocessed 밖 lot {len(miss)}개 → daily raw 캐시로 TRACE/TTV/AVE_THK 보충")
    rep = rep.merge(block_info, left_on=['EQP_ID', 'LOT_ID'],
                    right_on=['EQP_ID', 'BLK_NO'], how='left')
    rep = rep.drop(columns=['BLK_NO'])
    if len(rep):
        rep['data_src'] = np.where(rep['source'] == 'pilot', 'pilot', 'preprocessed')

    # 2b) 전체 블록(08-01~) 합치기 - Tuning_Group:
    #     APN_LOT의 **가장 과거 lot 이전** 블록 = 'Other', 이후에 APN_LOT에 없는 블록 = 'Eng'r 변경'
    #     (rep에 없는 블록만 추가; prof 컬럼을 rep 스키마에 맞춤, reindex로 열 정렬)
    out = rep
    other_n = eng_ext_n = 0
    cutoff = rep['TRACE_DTTS'].min() if len(rep) and 'TRACE_DTTS' in rep else None
    # 기존 csv의 APN_LOT 행(group_src='csv')까지 포함해 cutoff 고정
    # (pq가 20일 창이라 오래된 APN lot이 rep에서 빠져도 Other/Eng'r 경계가 밀리지 않게)
    old_blocks = _load_existing_blocks(BLOCKS_CSV)
    if len(old_blocks) and 'group_src' in old_blocks.columns:
        c_old = old_blocks.loc[old_blocks['group_src'] == 'csv', 'TRACE_DTTS'].min()
        if pd.notna(c_old):
            cutoff = c_old if (cutoff is None or pd.isna(cutoff)) else min(cutoff, c_old)
    if len(prof):
        rep_idx = rep.set_index(['EQP_ID', 'LOT_ID']).index
        o = prof[~prof.set_index(['EQP_ID', 'BLK_NO']).index.isin(rep_idx)]
        if len(o):
            o = o.copy()
            o = o.rename(columns={'BLK_NO': 'LOT_ID', 'avg_bow': 'actual_bow',
                                  'avg_warp': 'actual_warp'})
            o = _apply_pq_values(o, pq, ttvm)   # pq → by_eqp mean → raw 순
            vs = o['data_src'].value_counts().to_dict()
            print(f"   [2b] APN_LOT 밖 블록 값 출처: " + ' / '.join(f'{k} {v}' for k, v in vs.items()))
            if cutoff is not None:
                o['Tuning_Group'] = np.where(o['TRACE_DTTS'] < cutoff, OTHER_GROUP, ENG_GROUP)
            else:
                o['Tuning_Group'] = OTHER_GROUP
            o['group_src'] = 'all_raw'
            o = o.reindex(columns=rep.columns)
            out = pd.concat([rep, o], ignore_index=True)
            other_n = int((o['Tuning_Group'] == OTHER_GROUP).sum())
            eng_ext_n = int((o['Tuning_Group'] == ENG_GROUP).sum())
            print(f"   [전체] APN_LOT 밖 블록 {len(o)}개 추가 "
                  f"(cutoff={cutoff} 이전 Other {other_n} / 이후 Eng'r {eng_ext_n})")
    # 2c) pred 누락 블록은 모델 예측으로 보강 (feasible = 모델 있는 BSWS 13.3/18.5)
    #     roll=직전 10블럭 FDC median(_roll_at 동일), ref_frame=직전 blk(±0.5 clamp 재현)
    pred = build_predictions(prof) if len(prof) else pd.DataFrame()
    if len(pred):
        out = out.merge(pred.rename(columns={'BLK_NO': 'LOT_ID'}),
                        on=['EQP_ID', 'LOT_ID'], how='left', suffixes=('', '_pf'))
        for src in ('pred_rec_pf', 'pred_actual_pf'):
            dst = src[:-3]
            if src in out.columns:
                out[dst] = out[dst].fillna(out[src])
                out = out.drop(columns=[src])
    # err 재계산 (pred가 채워진 행 중 err 누락인 것)
    for pr, er in (('pred_rec', 'err_rec'), ('pred_actual', 'err_actual')):
        if er not in out.columns:
            out[er] = np.nan
        m = out[er].isna() & out[pr].notna() & out['actual_bow'].notna()
        out.loc[m, er] = (out.loc[m, 'actual_bow'] - out.loc[m, pr]).round(3)
    out = out.merge(load_eqp_nm(), on='EQP_ID', how='left')
    # 대상 필터: 31개 pilot 장비(EQP_NM 기준)만 유지 - EQP_ID가 아닌 **EQP_NM**으로.
    n_before = len(out)
    out = out[out['EQP_NM'].isin(EQUIP_FILTER)]
    out = out.sort_values('TRACE_DTTS', kind='stable').reset_index(drop=True)
    if len(out) < n_before:
        print(f"   [필터] EQP_NM 대상 {len(EQUIP_FILTER)}개 유지 → {len(out)}/{n_before}행 (제외 {n_before-len(out)}행)")
    # preprocessed에 있는 블록만 갱신/추가, 나머지 기존 행은 유지
    out, n_add, n_upd, n_keep = _merge_with_existing(out, old_blocks, pq)
    out = out.sort_values('TRACE_DTTS', kind='stable').reset_index(drop=True)
    print(f"   [병합] 신규 {n_add} / 갱신 {n_upd} / 기존 유지 {n_keep} → 총 {len(out)}행")

    os.makedirs(DATA_DIR, exist_ok=True)
    out.to_csv(BLOCKS_CSV, index=False, encoding='utf-8-sig')
    vc = out['Tuning_Group'].value_counts().to_dict()
    n_pred = int(out['pred_rec'].notna().sum())
    print(f" [2] 리포트 데이터: {len(out)}행 (APN_LOT {len(rep)} + Other {other_n} + Eng'r 확장 {eng_ext_n}) → {BLOCKS_CSV}")
    print(f"     csv 그룹: " + ' / '.join(f'{k} {v}' for k, v in vc.items()))
    print(f"     TRACE 범위: {out['TRACE_DTTS'].min()} ~ {out['TRACE_DTTS'].max()}   pred_rec 보유 {n_pred}행")
    return out


if __name__ == '__main__':
    main()
