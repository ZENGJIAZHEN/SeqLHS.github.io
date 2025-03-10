from flask import Flask, render_template, request, jsonify, redirect, url_for, session
import sqlite3
import bcrypt
from flask_session import Session
from flask import Flask, render_template, request, jsonify
import os
import io
import csv
import sys
import torch
import numpy as np
# 將 model 目錄加入 sys.path，確保可導入自訂的機器學習模組
sys.path.append(os.path.join(os.path.dirname(__file__), "model"))
# 匯入自訂模組
from generate_ud_samples import generate_ud_samples
from lhs_generator import generate_lhs_samples
from train_rsm import train_model as train_rsm_model
from train_mlp import train_mlp, predict_mlp, MLPRegressor
from train_rsm import predict as predict_rsm
from train_rsm import RSMModel, PolynomialFeatures
from analyze_data_pattern import analyze_data_pattern
from utils import evaluate_model_accuracy, calculate_interest_value, create_block_mapping
from sampling import create_latin_hypercube_sampling, update_search_space

app = Flask(__name__)
SAVE_DIR = "saved_projects"

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

# 模型存放資料夾
MODELS_DIR = "models" 
if not os.path.exists(MODELS_DIR):
    os.makedirs(MODELS_DIR)

app.secret_key = "super_secret_key"

# 設定 session 存儲方式
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "filesystem"
app.config["SESSION_COOKIE_NAME"] = "my_app_session"
Session(app)

# 建立 SQLite 資料庫 (如果不存在)
def init_db():
    with sqlite3.connect("users.db") as conn:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password TEXT NOT NULL
            )
        """)
        conn.commit()

init_db()

# 🔹 **首頁 (檢查登入)**
@app.route('/')
def index():
    if 'username' not in session:
        return redirect(url_for('login_page'))  # 未登入者導向登入頁面
    return render_template('index.html', username=session['username'])  # 登入後顯示主頁

# 🔹 **顯示登入頁面**
@app.route('/login', methods=['GET'])
def login_page():
    return render_template('login.html')  # 確保 `templates/login.html` 存在

# 🔹 **處理登入請求**
@app.route('/login', methods=['POST'])
def login():
    data = request.json
    username = data.get('username')
    password = data.get('password')

    with sqlite3.connect("users.db") as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT password FROM users WHERE username = ?", (username,))
        user = cursor.fetchone()

        if user and bcrypt.checkpw(password.encode('utf-8'), user[0].encode('utf-8')):  # 修正 bcrypt 檢查
            session['username'] = username  # 設定 session
            return jsonify({"success": True, "message": "登入成功", "redirect": url_for("index")})
        else:
            return jsonify({"success": False, "message": "帳號或密碼錯誤"}), 401

# 🔹 **註冊新帳戶**
@app.route('/register', methods=['POST'])
def register():
    data = request.json
    username = data.get('username')
    password = data.get('password')

    if not username or not password:
        return jsonify({"success": False, "message": "帳號或密碼不可為空"}), 400

    hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')  # 修正 bcrypt 存入問題

    try:
        with sqlite3.connect("users.db") as conn:
            cursor = conn.cursor()
            cursor.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, hashed_password))
            conn.commit()
        return jsonify({"success": True, "message": "註冊成功"})
    except sqlite3.IntegrityError:
        return jsonify({"success": False, "message": "帳號已存在"}), 400

# 🔹 **登出**
@app.route('/logout', methods=['GET'])
def logout():
    session.pop('username', None)
    return redirect(url_for('login_page'))  # 登出後回到登入畫面

# 處理上傳的 CSV 並回傳解析後的數據
@app.route('/train', methods=['POST'])
def train():
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    try:
        stream = io.StringIO(file.stream.read().decode("UTF8"), newline=None)
        csv_input = csv.reader(stream)
        
        headers = next(csv_input)
        data = list(csv_input)
        
        return jsonify({
            'message': 'File uploaded successfully',
            'headers': headers,
            'data': data
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# 儲存專案數據與上下界 CSV 檔案
@app.route('/save_csv', methods=['POST'])
def save_csv():
    try:
        project_name = request.form["project_name"].strip()
        project_dir = os.path.join(SAVE_DIR, project_name)

        if not os.path.exists(project_dir):
            os.makedirs(project_dir)

        csv_data = request.form["csv_data"]
        bounds_data = request.form["bounds_data"]

        csv_path = os.path.join(project_dir, f"{project_name}.csv")
        bounds_path = os.path.join(project_dir, f"{project_name}_bounds.csv")

        with open(csv_path, "w", newline="") as f:
            f.write(csv_data)

        with open(bounds_path, "w", newline="") as f:
            f.write(bounds_data)

        return jsonify({"message": "專案儲存成功"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# 回傳已儲存的專案清單
@app.route('/list_projects')
def list_projects():
    projects = [name for name in os.listdir(SAVE_DIR) if os.path.isdir(os.path.join(SAVE_DIR, name))]
    return jsonify(projects)

# 讀取指定的專案數據與上下界
@app.route('/load_csv')
def load_csv():
    try:
        project_name = request.args.get("project_name").strip()
        project_dir = os.path.join(SAVE_DIR, project_name)

        csv_path = os.path.join(project_dir, f"{project_name}.csv")
        bounds_path = os.path.join(project_dir, f"{project_name}_bounds.csv")

        if not os.path.exists(csv_path) or not os.path.exists(bounds_path):
            return jsonify({"error": "專案不存在"}), 404

        with open(csv_path, newline="") as f:
            reader = csv.reader(f)
            headers = next(reader)
            data = list(reader)

        with open(bounds_path, newline="") as f:
            reader = csv.reader(f)
            bounds = list(reader)[1:]  # 跳過標題行，只取數據

        return jsonify({"headers": headers, "data": data, "bounds": bounds})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/generate_samples', methods=['POST'])
def generate_samples():
    try:
        data = request.get_json()
        num_samples = data["num_samples"]
        bounds = np.array(data["bounds"])

        samples = generate_ud_samples(num_samples, bounds)

        return jsonify({"samples": samples.tolist()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# 使用 LHS 生成樣本    
@app.route('/generate_lhs', methods=['POST'])
def generate_lhs():
    try:
        data = request.json
        num_samples = data.get("num_samples")
        bounds = data.get("bounds")

        if not isinstance(num_samples, int) or num_samples <= 0:
            return jsonify({"error": "樣本數量無效"}), 400
        if not isinstance(bounds, list) or len(bounds) == 0:
            return jsonify({"error": "上下界無效"}), 400

        for bound in bounds:
            if bound is None or not isinstance(bound, list) or len(bound) != 2:
                return jsonify({"error": "上下界格式錯誤"}), 400
            if bound[0] is None or bound[1] is None or not isinstance(bound[0], (int, float)) or not isinstance(bound[1], (int, float)):
                return jsonify({"error": "上下界值錯誤"}), 400
            if bound[0] >= bound[1]:
                return jsonify({"error": "下界必須小於上界"}), 400

        # 呼叫 LHS 生成函式
        samples = generate_lhs_samples(num_samples, bounds)
        return jsonify({"samples": samples})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/project_exists', methods=['GET'])
def project_exists():
    project_name = request.args.get("name", "").strip()
    if not project_name:
        return jsonify({"exists": False})
    project_dir = os.path.join(SAVE_DIR, project_name)
    return jsonify({"exists": os.path.isdir(project_dir)})

@app.route('/save_project', methods=['POST'])
def save_project():
    try:
        project_name = request.form.get("project_name", "").strip()
        points_csv = request.form.get("points_csv", "")
        bounds_csv = request.form.get("bounds_csv", "")

        if not project_name:
            return jsonify({"error": "專案名稱不能為空"}), 400

        project_path = os.path.join(SAVE_DIR, project_name)
        os.makedirs(project_path, exist_ok=True)
        with open(os.path.join(project_path, "points.csv"), "w", encoding="utf-8-sig", newline="") as f:
            f.write(points_csv)
    
        with open(os.path.join(project_path, "bounds.csv"), "w", encoding="utf-8-sig", newline="") as f:
            f.write(bounds_csv)

        with open(os.path.join(project_path, "points.csv"), "w", encoding="utf-8-sig", newline="") as f:
            f.write(points_csv)

        with open(os.path.join(project_path, "bounds.csv"), "w", encoding="utf-8-sig", newline="") as f:
            f.write(bounds_csv)

        return jsonify({"message": f"專案 {project_name} 已成功儲存！"}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# 載入專案
@app.route("/load_project", methods=["GET"])
def load_project():
    try:
        project_name = request.args.get("project_name","").strip()
        if not project_name:
            return jsonify({"error":"no project_name"}), 400

        project_dir = os.path.join(SAVE_DIR, project_name)
        points_path = os.path.join(project_dir, "points.csv")
        bounds_path = os.path.join(project_dir, "bounds.csv")
        if not (os.path.exists(points_path) and os.path.exists(bounds_path)):
            return jsonify({"error":f"專案「{project_name}」不存在"}), 404

        with open(points_path, "r", encoding="utf-8-sig") as f:
            points_csv_content = f.read()
        with open(bounds_path, "r", encoding="utf-8-sig") as f:
            bounds_csv_content = f.read()

        return jsonify({
            "pointsCsv": points_csv_content,
            "boundsCsv": bounds_csv_content
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# 訓練模型    
@app.route("/train_model", methods=["POST"])
def train_model_api():
    try:
        data = request.get_json()
        model_type = data["model_type"]
        table_data = data["data"]
        
        table_data = np.array(table_data)
        X = table_data[:, :-1]  
        y = table_data[:, -1]   

        # 統一使用 model.pt
        model_path = os.path.join(MODELS_DIR, "model.pt")
        
        if model_type == "RSM":
            rsm_result = train_rsm_model(X, y)
            r2 = rsm_result["r2"]
            
            model_state = {
                "model_type": "RSM",  # 加入模型類型標記
                "state_dict": rsm_result["model"].state_dict(),
                "poly_degree": rsm_result["degree"],
                "r2": rsm_result["r2"],
                "clusters": rsm_result["clusters"],
                "poly_powers": rsm_result["poly"].powers_,
                "n_output_features": rsm_result["poly"].n_output_features_,
                "input_dim": X.shape[1]
            }
            msg = f"RSM 訓練完成 (自動偵測 degree 與 clusters), R²={r2:.4f}"
        
        else:  # MLP
            hidden_layers = data.get("hidden_layers", [3])
            activation = data.get("activation", "relu")
            max_iter = data.get("max_iter", 2000)
            learning_rate = data.get("learning_rate", 0.001)
            random_state = data.get("random_state", 42)
            batch_size = data.get("batch_size", 32)
            
            mlp_model, r2_score, loss, needs_param_change = train_mlp(
                X, y,
                hidden_layers=hidden_layers,
                activation=activation,
                max_iter=max_iter,
                learning_rate=learning_rate,
                random_state=random_state,
                batch_size=batch_size
            )
            r2 = r2_score

            model_state = {
                "model_type": "MLP",  # 加入模型類型標記
                "state_dict": mlp_model.state_dict(),
                "hidden_layers": hidden_layers,
                "activation": activation,
                "r2_score": r2_score,
                "loss": loss,
                "input_dim": X.shape[1]
            }
            msg = f"MLP 訓練完成 R²={r2:.4f}, Loss={loss:.4f}"

        # 儲存模型
        torch.save(model_state, model_path)

        return jsonify({
            "message": msg,
            "model_type": model_type,
            "rows": len(X),
            "r2": float(r2)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# 載入模型狀態
def load_model_state():
    model_path = os.path.join(MODELS_DIR, "model.pt")
    if not os.path.exists(model_path):
        raise FileNotFoundError("找不到已訓練的模型")
    
    model_state = torch.load(model_path, weights_only=False)
    model_type = model_state["model_type"]
    
    if model_type == "RSM":
        # 重建 RSM 模型
        model = RSMModel(model_state["n_output_features"])
        model.load_state_dict(model_state["state_dict"])
        
        # 重建 PolynomialFeatures
        poly = PolynomialFeatures(model_state["poly_degree"])
        poly.powers_ = model_state["poly_powers"]
        poly.n_output_features_ = model_state["n_output_features"]
        
        return {
            "model": model,
            "poly": poly,
            "degree": model_state["poly_degree"],
            "r2": model_state["r2"],
            "clusters": model_state["clusters"]
        }
    else:  # MLP
        # 重建 MLP 模型
        model = MLPRegressor(
            input_size=model_state["input_dim"],
            hidden_layers=model_state["hidden_layers"],
            activation=model_state["activation"]
        )
        model.load_state_dict(model_state["state_dict"])
        
        return model

# 生成預測新點位
@app.route("/generate_new_points", methods=["POST"])
def generate_new_points():
    try:
        data = request.get_json()
        current_data = np.array(data["current_data"])
        bounds = np.array(data["bounds"])
        num_points = data["num_points"]

        current_points = current_data[:, :-1]
        current_values = current_data[:, -1]

        new_bounds, new_divisions = update_search_space(
            current_points, 
            current_values, 
            bounds, 
            num_points,
            d_reduction=1
        )

        try:
            loaded_model = load_model_state()
        except FileNotFoundError:
            return jsonify({"error": "找不到已訓練的模型"}), 404

        interest_values = calculate_interest_value(
            current_points,
            current_values,
            loaded_model,
            new_bounds,
            new_divisions
        )

        block_mapping = create_block_mapping(
            interest_values,
            new_bounds,
            new_divisions
        )

        new_points = create_latin_hypercube_sampling(
            block_mapping,
            num_points,
            new_bounds,
            current_points
        )

        return jsonify({
            "new_points": new_points.tolist(),
            "new_bounds": new_bounds.tolist(),
            "message": "成功生成新點位"
        })
    except Exception as e:
        print(f"Error in generate_new_points: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
    
if __name__ == '__main__':
    app.run(debug=True)
