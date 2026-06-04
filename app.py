import streamlit as st
import pandas as pd
import numpy as np
import os
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 1. Cấu hình giao diện trang web
st.set_page_config(page_title="Hệ thống Thẩm định & Phát hiện Gian lận Tài chính AI", layout="wide")

# Khởi tạo trạng thái đăng nhập hệ thống
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False
if 'users' not in st.session_state:
    st.session_state['users'] = {'admin': 'admin123'}

def login_page():
    st.title("🔐 ĐĂNG NHẬP HỆ THỐNG")
    tab1, tab2 = st.tabs(["Đăng nhập", "Đăng ký tài khoản"])
    
    with tab1:
        username = st.text_input("Tên đăng nhập:", key="login_user")
        password = st.text_input("Mật khẩu:", type="password", key="login_pass")
        if st.button("Đăng nhập 🚀", use_container_width=True):
            if username in st.session_state['users'] and st.session_state['users'][username] == password:
                st.session_state['logged_in'] = True
                st.session_state['current_user'] = username
                st.success("Đăng nhập thành công!")
                st.rerun()
            else:
                st.error("Sai tên đăng nhập hoặc mật khẩu!")
                
    with tab2:
        new_user = st.text_input("Tạo tài khoản mới:", key="reg_user")
        new_pass = st.text_input("Tạo mật khẩu:", type="password", key="reg_pass")
        confirm_pass = st.text_input("Xác nhận mật khẩu:", type="password", key="reg_confirm")
        if st.button("Đăng ký tài khoản 📝", use_container_width=True):
            if not new_user or not new_pass:
                st.warning("Vui lòng không để trống thông tin!")
            elif new_pass != confirm_pass:
                st.error("Mật khẩu xác nhận không khớp!")
            elif new_user in st.session_state['users']:
                st.error("Tài khoản đã tồn tại!")
            else:
                st.session_state['users'][new_user] = new_pass
                st.success("Đăng ký thành công! Hãy chuyển sang tab Đăng nhập.")

# CHẠY KIỂM TRA ĐĂNG NHẬP
if not st.session_state['logged_in']:
    login_page()
else:
    # --- GIAO DIỆN CHÍNH ---
    st.sidebar.title(f"👤 Xin chào, {st.session_state.get('current_user', 'User')}")
    if st.sidebar.button("🚪 Đăng xuất"):
        st.sidebar.session_state['logged_in'] = False
        st.rerun()
        
    st.sidebar.markdown("---")
    st.sidebar.subheader("📁 Quản lý Dữ liệu")
    
    # BỔ SUNG: Chức năng tải file CSV từ máy tính
    uploaded_file = st.sidebar.file_uploader("Tải lên file dữ liệu của bạn (.CSV)", type=["csv"])
    
    DATA_FILE = "dữ liệu lớn.csv"
    
    # Cập nhật hàm load dữ liệu để ưu tiên đọc file vừa tải lên
    @st.cache_data
    def load_data(file_obj):
        if file_obj is not None:
            df = pd.read_csv(file_obj)
            st.sidebar.success("✅ Đã nạp dữ liệu từ file bạn tải lên!")
        elif os.path.exists(DATA_FILE):
            df = pd.read_csv(DATA_FILE)
            st.sidebar.success(f"✅ Đã nạp dữ liệu gốc từ: `{DATA_FILE}`")
        else:
            st.sidebar.warning(f"⚠️ Đang chạy dữ liệu giả lập dự phòng (Không tìm thấy dữ liệu đầu vào).")
            np.random.seed(42)
            rows = 200
            df = pd.DataFrame({
                'Total_Assets': np.random.uniform(0.5, 5.0, rows),
                'Total_Liabilities': np.random.uniform(0.2, 4.0, rows),
                'Revenue': np.random.uniform(1.0, 10.0, rows),
                'Operating_Expenses': np.random.uniform(0.5, 4.0, rows),
                'Net_Income': np.random.uniform(-0.5, 1.5, rows),
                'Cash_Flow_Operating': np.random.uniform(-0.8, 2.0, rows),
                'Current_Ratio': np.random.uniform(0.5, 3.0, rows),
                'Debt_to_Equity': np.random.uniform(0.1, 5.0, rows),
                'Gross_Margin': np.random.uniform(-0.2, 0.8, rows),
                'Return_on_Assets': np.random.uniform(-0.1, 0.3, rows),
                'Return_on_Equity': np.random.uniform(-0.2, 0.5, rows),
                'Financial_Status': np.random.choice(['Normal', 'Fraud'], size=rows)
            })
            
        if 'Nam' not in df.columns and 'Year' not in df.columns:
            df['Year'] = np.random.choice([2023, 2024, 2025, 2026], size=len(df))
        return df

    # Gọi hàm load data với file object truyền vào
    df = load_data(uploaded_file)
    
    # Danh sách các nhãn rủi ro mở rộng (tương thích với cả file mới của bạn)
    FRAUD_LABELS = [1, '1', 'Fraud', 'fraud', 'Abnormal', 'High Risk', 'Anomaly']
    
    @st.cache_resource
    def train_ai_model(X_data, y_data):
        X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, random_state=42, stratify=y_data)
        model = RandomForestClassifier(random_state=42, n_estimators=100)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0)
        }
        return model, metrics

    # --- MENU ĐIỀU HƯỚNG ---
    menu = st.sidebar.selectbox("Chức năng hệ thống", [
        "🏠 Trang chủ & Tổng quan dữ liệu",
        "📊 Phân tích Chỉ số Tài chính chuyên sâu",
        "⚠️ Quét Dấu hiệu Rủi ro & Gian lận",
        "🤖 Mô hình AI Dự báo Gian lận",
        "📉 Báo cáo Trực quan (Biểu đồ xu hướng)",
        "📋 Kết xuất Báo cáo Giám định"
    ])
    
    # 1. TRANG CHỦ
    if menu == "🏠 Trang chủ & Tổng quan dữ liệu":
        st.title("🏠 Hệ thống Giám định & Phân tích Gian lận Tài chính")
        st.write("Dữ liệu được nạp tự động, chuẩn hóa cấu trúc phân tích.")
        
        st.subheader("📊 Thông số tóm tắt tập dữ liệu")
        col1, col2, col3 = st.columns(3)
        col1.metric("Tổng số bản ghi (Doanh nghiệp)", f"{len(df)} dòng")
        col2.metric("Số lượng biến số phân tích", f"{len(df.columns)} cột")
        
        target_col = 'Financial_Status' if 'Financial_Status' in df.columns else df.columns[-1]
        fraud_count = int(df[target_col].isin(FRAUD_LABELS).sum())
        col3.metric("Số hồ sơ Cảnh báo Rủi ro", f"{fraud_count} mục")
        
        st.subheader("📁 Xem trước cấu trúc bảng dữ liệu (10 dòng đầu)")
        st.dataframe(df.head(10))

    # 2. PHÂN TÍCH CHỈ SỐ CHUYÊN SÂU
    elif menu == "📊 Phân tích Chỉ số Tài chính chuyên sâu":
        st.header("📊 Thống kê Mô tả Sức khỏe Tài chính")
        desc_cols = [c for c in ['Revenue', 'Net_Income', 'Current_Ratio', 'Debt_to_Equity', 'Return_on_Assets', 'Return_on_Equity'] if c in df.columns]
        if desc_cols:
            st.dataframe(df[desc_cols].describe().T)
        else:
            st.dataframe(df.select_dtypes(include=[np.number]).describe().T)

    # 3. QUÉT DẤU HIỆU RỦI RO
    elif menu == "⚠️ Quét Dấu hiệu Rủi ro & Gian lận":
        st.header("⚠️ Bộ quy tắc Quét Điểm Bất thường Hệ thống")
        t1, t2 = st.tabs(["🚨 Dòng tiền lệch pha với Lợi nhuận", "📉 Khả năng thanh toán nguy hiểm"])
        
        with t1:
            if 'Net_Income' in df.columns and 'Cash_Flow_Operating' in df.columns:
                dh1 = df[(df['Net_Income'] > 0) & (df['Cash_Flow_Operating'] < 0)]
                st.warning(f"Phát hiện {len(dh1)} trường hợp Lợi nhuận ròng dương nhưng Dòng tiền HĐKD âm.")
                st.dataframe(dh1)
            else:
                st.info("Không tìm thấy trường dữ liệu 'Net_Income' và 'Cash_Flow_Operating'.")
                
        with t2:
            if 'Current_Ratio' in df.columns and 'Debt_to_Equity' in df.columns:
                dh2 = df[(df['Current_Ratio'] < 1.0) & (df['Debt_to_Equity'] > 2.0)]
                st.error(f"Phát hiện {len(dh2)} đối tượng rủi ro mất khả năng thanh toán.")
                st.dataframe(dh2)
            else:
                st.info("Không tìm thấy trường dữ liệu 'Current_Ratio' và 'Debt_to_Equity'.")

    # 4. MÔ HÌNH AI DỰ BÁO GIAN LẬN
    elif menu == "🤖 Mô hình AI Dự báo Gian lận":
        st.header("🤖 Học máy AI (Random Forest Classifier) phân tích đa chiều")
        
        target_col = 'Financial_Status' if 'Financial_Status' in df.columns else df.columns[-1]
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        for c in ['Nam', 'Year', target_col]:
            if c in numeric_cols: numeric_cols.remove(c)
            
        if len(numeric_cols) >= 2:
            X = df[numeric_cols].fillna(df[numeric_cols].median())
            
            if df[target_col].dtype == 'object':
                y = np.where(df[target_col].isin(FRAUD_LABELS), 1, 0)
            else:
                y = df[target_col].fillna(0).astype(int)
                
            if len(np.unique(y)) < 2:
                y[0] = 1 - y[0] 
                
            clf, model_metrics = train_ai_model(X, y)
            
            st.success("🚀 Mô hình học máy AI đã đọc dữ liệu đầu vào và huấn luyện thành công!")
            
            st.subheader("🎯 Chỉ số đánh giá chất lượng thuật toán học máy (Validation Metrics)")
            m_col1, m_col2, m_col3, m_col4 = st.columns(4)
            m_col1.metric("Độ chính xác tổng thể (Accuracy)", f"{model_metrics['accuracy']*100:.2f}%")
            m_col2.metric("Độ chuẩn xác dự báo (Precision)", f"{model_metrics['precision']*100:.2f}%")
            m_col3.metric("Khả năng bắt sót rủi ro (Recall)", f"{model_metrics['recall']*100:.2f}%")
            m_col4.metric("Điểm F1-Score cân bằng", f"{model_metrics['f1']*100:.2f}%")
            
            st.markdown("---")
            st.subheader("🕵️ Kiểm định hồ sơ doanh nghiệp đơn lẻ")
            input_data = []
            show_inputs = [c for c in ['Revenue', 'Net_Income', 'Return_on_Assets'] if c in numeric_cols]
            if len(show_inputs) < 3:
                show_inputs = numeric_cols[:3]
                
            cols = st.columns(len(show_inputs))
            for idx, col_name in enumerate(show_inputs):
                with cols[idx]:
                    val = st.number_input(f"Chỉ số {col_name}:", value=float(df[col_name].mean()))
                    input_data.append(val)
                    
            if st.button("🚀 Kích Hoạt AI Kiểm Định"):
                full_input = []
                for c in numeric_cols:
                    if c in show_inputs:
                        full_input.append(input_data[show_inputs.index(c)])
                    else:
                        full_input.append(float(df[c].mean()))
                        
                input_values = np.array([full_input]).reshape(1, -1)
                
                pred = clf.predict(input_values)[0]
                proba_matrix = clf.predict_proba(input_values)[0]
                prob = proba_matrix[1] if len(proba_matrix) > 1 else (1.0 if pred == 1 else 0.0)
                
                st.markdown("---")
                if pred == 1:
                    st.error(f"🚨 CẢNH BÁO: Thực thể này có dấu hiệu rủi ro cao! Xác suất bất thường: {prob*100:.2f}%")
                else:
                    st.success(f"✅ AN TOÀN: Chỉ số phân phối bình thường. Xác suất rủi ro: {prob*100:.2f}%")
        else:
            st.warning("Cần tối thiểu 2 cột dữ liệu định dạng số để huấn luyện mô hình AI.")

    # 5. BÁO CÁO TRỰC QUAN
    elif menu == "📉 Báo cáo Trực quan (Biểu đồ xu hướng)":
        st.header("📉 Trực quan hóa Biến động Chỉ số Tổng cục")
        group_col = 'Nam' if 'Nam' in df.columns else 'Year'
        df_year = df.groupby(group_col).mean(numeric_only=True).reset_index()
        
        c1, c2 = st.columns(2)
        with c1:
            y_axis = ['Revenue', 'Net_Income'] if 'Revenue' in df.columns and 'Net_Income' in df.columns else df_year.columns[1:3].tolist()
            fig1 = px.line(df_year, x=group_col, y=y_axis, title="Xu hướng biến động Doanh thu & Lợi nhuận bình quân", markers=True)
            st.plotly_chart(fig1, use_container_width=True)
        with c2:
            target_y = 'Return_on_Assets' if 'Return_on_Assets' in df.columns else df_year.columns[2]
            fig2 = px.bar(df_year, x=group_col, y=target_y, title="Tỷ suất sinh lời bình quân trên tài sản (ROA)", color=target_y)
            st.plotly_chart(fig2, use_container_width=True)

    # 6. KẾT XUẤT BÁO CÁO
    elif menu == "📋 Kết xuất Báo cáo Giám định":
        st.header("📋 Tổng hợp Kết quả Thẩm định Hệ thống")
        total_rec = len(df)
        target_col = 'Financial_Status' if 'Financial_Status' in df.columns else df.columns[-1]
        fraud_rec = int(df[target_col].isin(FRAUD_LABELS).sum())
        risk_percentage = (fraud_rec / total_rec) * 100 if total_rec > 0 else 0
        
        current_source = uploaded_file.name if uploaded_file is not None else DATA_FILE
        
        report_preview = f"""==================================================
BÁO CÁO GIÁM ĐỊNH RỦI RO GIAN LẬN TÀI CHÍNH TOÀN CỤC
==================================================
- Thẩm định viên thực hiện: {st.session_state.get('current_user', 'Hệ thống AI')}
- Tổng số lượng doanh nghiệp/hồ sơ đã phân tích: {total_rec} đối tượng.
- Tổng số lượng thực thể dính nhãn cảnh báo rủi ro: {fraud_rec} đối tượng.
- Tỷ lệ rủi ro hệ thống: {risk_percentage:.2f}%
--------------------------------------------------
Nguồn dữ liệu kết xuất tự động từ tệp tin: {current_source}
"""
        st.text_area("Văn bản báo cáo tổng hợp tóm tắt:", report_preview, height=220)
        csv_buffer = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Tải xuống Toàn bộ Bảng dữ liệu đã xử lý (.CSV)",
            data=csv_buffer,
            file_name="Bao_cao_giam_dinh_tai_chinh.csv",
            mime="text/csv",
            use_container_width=True
        )