"""
KCET College Predictor — Flask API Server
"""

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
import pandas as pd
import json
import sys
import os
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR / 'src'))

from predict import predict, load_assets, get_trends

app = Flask(__name__, static_folder=str(BASE_DIR / 'frontend'))
CORS(app)


# ── Static frontend ────────────────────────────────────────────────────────
@app.route('/')
def index():
    return send_from_directory(app.static_folder, 'index.html')


# ── API: predict colleges ──────────────────────────────────────────────────
@app.route('/api/predict')
def api_predict():
    try:
        rank = int(request.args.get('rank', 0))
        category = request.args.get('category', '1G').strip()
        branches_raw = request.args.get('branches', '')
        top_n = int(request.args.get('top_n', 150))

        branch_filter = [b.strip() for b in branches_raw.split(',') if b.strip()] \
                        if branches_raw else None

        if rank <= 0:
            return jsonify({'error': 'Invalid rank'}), 400

        results = predict(user_rank=rank, category=category,
                          branch_filter=branch_filter, top_n=top_n)
        return jsonify({'results': results, 'count': len(results)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ── API: metadata ──────────────────────────────────────────────────────────
@app.route('/api/meta')
def api_meta():
    _, colleges, branches, categories, _ = load_assets()
    return jsonify({
        'colleges':   colleges.to_dict(orient='records'),
        'branches':   branches['branch_name'].tolist(),
        'categories': categories,
    })


# ── API: trend for a specific combo ───────────────────────────────────────
@app.route('/api/trend')
def api_trend():
    college_code = request.args.get('college', '')
    branch       = request.args.get('branch', '')
    category     = request.args.get('category', '1G')
    data = get_trends(college_code, branch, category)
    return jsonify(data)


# ── API: health check ─────────────────────────────────────────────────────
@app.route('/api/health')
def health():
    return jsonify({'status': 'ok'})


if __name__ == '__main__':
    print("Starting KCET College Predictor server...")
    print("   Open: http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=False)
