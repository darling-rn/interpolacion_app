import streamlit as st
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d, CubicSpline
import matplotlib.pyplot as plt
from PIL import Image # Para la aplicación de procesamiento de imágenes

# --- Configuración de la Página ---
st.set_page_config(
    page_title="Aplicaciones de Interpolación",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS Personalizado para Estilo ---
st.markdown("""
    <style>
    .main-header {
        font-size: 3em;
        color: #4CAF50; /* Verde vibrante */
        text-align: center;
        margin-bottom: 30px;
        font-family: 'Arial Black', Gadget, sans-serif;
    }
    .subheader {
        font-size: 2em;
        color: #2196F3; /* Azul distintivo */
        margin-top: 20px;
        margin-bottom: 15px;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .stButton>button {
        background-color: #008CBA; /* Azul medio */
        color: white;
        padding: 10px 20px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        margin: 4px 2px;
        cursor: pointer;
        border-radius: 8px;
        border: none;
        transition: background-color 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #005f7a; /* Azul más oscuro al pasar el ratón */
    }
    .stTextInput>div>div>input, .stNumberInput>div>div>input {
        border-radius: 8px;
        border: 1px solid #ddd; /* Borde gris suave */
        padding: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1); /* Sombra sutil */
    }
    .info-block {
        background-color: #e3f2fd; /* Azul claro */
        border-left: 5px solid #2196F3; /* Borde izquierdo azul */
        padding: 15px;
        margin-bottom: 20px;
        border-radius: 5px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.05);
    }
    .stTabs [data-baseweb="tab-list"] button {
        background-color: #f0f0f0; /* Pestañas claras */
        color: #555;
        font-weight: bold;
        border-top-left-radius: 8px;
        border-top-right-radius: 8px;
        border: 1px solid #ccc;
        border-bottom: none;
        margin-right: 5px;
        padding: 10px 15px;
        transition: background-color 0.3s ease;
    }
    .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {
        background-color: #4CAF50; /* Pestaña activa verde */
        color: white;
        border-color: #4CAF50;
        box-shadow: 0 -2px 5px rgba(0,0,0,0.1);
    }
    .stProgress > div > div > div > div {
        background-color: #4CAF50; /* Color de la barra de progreso */
    }
    </style>
    """, unsafe_allow_html=True)

# --- Encabezado Principal ---
st.markdown("<h1 class='main-header'>Interpolación: Aplicaciones en el Mundo Real</h1>", unsafe_allow_html=True)
st.markdown("<h3>Explora cómo la interpolación nos ayuda a estimar datos en diversos campos.</h3>", unsafe_allow_html=True)

# --- Navegación en la Barra Lateral ---
st.sidebar.title("Navegación")
page = st.sidebar.radio("Ir a", ["Sobre Interpolación", "Pronóstico del Tiempo", "Procesamiento de Imágenes", "Modelado Financiero"])

# --- Página: Sobre Interpolación ---
if page == "Sobre Interpolación":
    st.markdown("<h2 class='subheader'>Sobre Interpolación</h2>", unsafe_allow_html=True)
    st.markdown("""
    <div class="info-block">
    La Interpolación es un método para construir nuevos puntos de datos dentro del rango de un conjunto discreto de puntos de datos conocidos. 
    Esencialmente, se trata de "leer entre líneas" en tus datos. Si tienes algunas mediciones en puntos específicos, 
    la interpolación te permite estimar cuál podría ser el valor en cualquier punto intermedio entre esas mediciones.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<h3>¿Por qué es importante?</h3>")
    st.write("""
    Muchos fenómenos del mundo real son continuos, pero solo podemos recolectar puntos de datos discretos. 
    La interpolación nos ayuda a:
    * Completar datos faltantes.
    * Suavizar datos ruidosos.
    * Remuestrear datos a una resolución diferente.
    * Hacer predicciones o estimaciones.
    """)

    st.markdown("<h3>Métodos Comunes de Interpolación:</h3>")
    
    tab1, tab2, tab3 = st.tabs(["Interpolación Lineal", "Interpolación Polinomial", "Interpolación por Splines"])

    with tab1:
        st.markdown("""
        **Concepto:** Conecta dos puntos de datos conocidos con una línea recta.

        **Fórmula:** $y = y_0 + (x - x_0) \\frac{y_1 - y_0}{x_1 - x_0}$

        **Pros:** Simple, computacionalmente económica.

        **Contras:** No es precisa para relaciones no lineales, produce esquinas agudas en los puntos de datos.
        """)
    
    with tab2:
        st.markdown("""
        **Concepto:** Ajusta una única curva polinomial que pasa por todos los puntos de datos conocidos.
        
        **Fórmula (Lagrange):** $P(x) = \sum_{j=0}^{n} y_j L_j(x)$
        **, donde**  $L_j(x) = \prod_{k=0, k \ne j}^{n} \\frac{x - x_k}{x_j - x_k}$
        
        **Pros:** Puede ser muy precisa si la función subyacente se aproxima bien con un polinomio.
        
        **Contras:** Polinomios de alto grado pueden llevar a oscilaciones (fenómeno de Runge) entre los puntos de datos, especialmente en los extremos.
        """)

    with tab3:
        st.markdown("""
        **Concepto:** Divide los datos en segmentos y ajusta un polinomio de bajo grado (a menudo cúbico) a cada segmento. Asegura la suavidad (continuidad de las derivadas) en los límites de los segmentos.
        
        **Pros:** Más estable que la interpolación polinomial de alto grado, maneja mejor las oscilaciones, produce curvas suaves. Ampliamente utilizada en la práctica.
        
        **Contras:** Más intensiva computacionalmente que la interpolación lineal.
        """)

# --- Aplicación 1: Pronóstico del Tiempo ---
elif page == "Pronóstico del Tiempo":
    st.markdown("<h2 class='subheader'>Aplicación 1: Pronóstico del Tiempo (Predicción de Temperatura)</h2>", unsafe_allow_html=True)
    st.markdown("""
    <div class="info-block">
    Imagina que tienes lecturas de temperatura cada hora, pero necesitas saber la temperatura en un minuto específico. 
    La interpolación puede ayudarte a estimar este valor.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<h4>Ingresa tus Puntos de Datos:</h4>")
    # Datos de ejemplo (pueden ser modificados por el usuario)
    default_times = [0, 1, 2, 3, 4, 5]
    default_temps = [10, 12, 15, 13, 11, 9]

    # Entrada para el número de puntos de datos
    num_points = st.number_input("Número de puntos de datos conocidos:", min_value=2, value=len(default_times), step=1)
    
    cols_input = st.columns(2)
    times = []
    temps = []

    st.write("Ingresa Hora (ej. 0 para 12 AM, 1 para 1 AM) y Temperatura:")
    for i in range(num_points):
        with cols_input[0]:
            t = st.number_input(f"Hora {i+1}:", value=float(default_times[i]) if i < len(default_times) else 0.0, key=f"time_{i}")
            times.append(t)
        with cols_input[1]:
            temp = st.number_input(f"Temperatura {i+1}:", value=float(default_temps[i]) if i < len(default_temps) else 0.0, key=f"temp_{i}")
            temps.append(temp)
    
    x_known = np.array(times)
    y_known = np.array(temps)

    st.markdown("<h4>Predice la Temperatura en un Momento Específico:</h4>")
    if len(x_known) > 1: # Asegurarse de que haya al menos dos puntos para interpolar
        predict_time = st.slider("Hora a Predecir (ej. 2.5 para 2:30 AM):", min_value=float(min(x_known)), max_value=float(max(x_known)), value=(min(x_known) + max(x_known)) / 2)
    else:
        st.warning("Necesitas al menos dos puntos de datos para realizar la interpolación.")
        predict_time = None

    if st.button("Calcular Interpolación") and predict_time is not None:
        st.markdown("<h4>Resultados:</h4>")

        # Interpolación Lineal
        f_linear = interp1d(x_known, y_known, kind='linear')
        predicted_linear = f_linear(predict_time)
        st.write(f"**Interpolación Lineal:** La temperatura predicha a la hora {predict_time:.2f} es **{predicted_linear:.2f}°C**")

        # Interpolación Polinomial (Lagrange)
        f_poly = None
        if len(x_known) >= 2: # Se necesita al menos 2 puntos para un polinomio de grado 1
            try:
                poly_coeffs = np.polyfit(x_known, y_known, len(x_known) - 1)
                f_poly = np.poly1d(poly_coeffs)
                predicted_poly = f_poly(predict_time)
                st.write(f"**Interpolación Polinomial (Lagrange):** La temperatura predicha a la hora {predict_time:.2f} es **{predicted_poly:.2f}°C**")
            except np.linalg.LinAlgError:
                st.warning("No se puede realizar la interpolación polinomial con estos puntos de datos (ej. muy pocos puntos o valores X no únicos).")
        else:
            st.warning("No hay suficientes puntos para la interpolación polinomial.")


        # Interpolación por Spline Cúbico
        if len(x_known) >= 2: # Se necesita al menos 2 puntos para spline (scipy)
            f_spline = CubicSpline(x_known, y_known)
            predicted_spline = f_spline(predict_time)
            st.write(f"**Interpolación por Spline Cúbico:** La temperatura predicha a la hora {predict_time:.2f} es **{predicted_spline:.2f}°C**")
        else:
            st.warning("No hay suficientes puntos para la interpolación por spline cúbico.")


        st.markdown("<h4>Comparación Gráfica:</h4>")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(x_known, y_known, 'o', label='Puntos de Datos Conocidos', markersize=8, color='red')

        x_interp = np.linspace(min(x_known), max(x_known), 500)
        ax.plot(x_interp, f_linear(x_interp), label='Interpolación Lineal', linestyle='--', color='blue')
        
        if f_poly is not None: 
            ax.plot(x_interp, f_poly(x_interp), label='Interpolación Polinomial', linestyle=':', color='green')
        
        if len(x_known) >= 2: # Solo si hay suficientes puntos
            ax.plot(x_interp, f_spline(x_interp), label='Interpolación por Spline Cúbico', linestyle='-.', color='purple')

        ax.axvline(predict_time, color='gray', linestyle=':', label=f'Predicción en {predict_time:.2f}')
        
        ax.set_xlabel("Hora (horas)")
        ax.set_ylabel("Temperatura (°C)")
        ax.set_title("Predicción de Temperatura usando Interpolación")
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)

# --- Aplicación 2: Procesamiento de Imágenes ---
elif page == "Procesamiento de Imágenes":
    st.markdown("<h2 class='subheader'>Aplicación 2: Procesamiento de Imágenes (Escalado de Imágenes)</h2>", unsafe_allow_html=True)
    st.markdown("""
    <div class="info-block">
    Cuando haces zoom en una imagen, se necesitan crear nuevos valores de píxeles. La interpolación ayuda 
    a estimar estos nuevos valores para que la imagen escalada se vea suave y natural.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<h4>Sube una Imagen Pequeña:</h4>")
    uploaded_file = st.file_uploader("Elige una imagen...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Imagen Original", use_column_width=True)

        st.markdown("<h4>Opciones de Escalado:</h4>")
        scale_factor = st.slider("Factor de Escala (ej. 2.0 para el doble de tamaño):", min_value=0.5, max_value=5.0, value=2.0, step=0.1)

        interpolation_method = st.selectbox(
            "Método de Interpolación para Escalado:",
            ("Nearest Neighbor", "Bilinear") # SciPy tiene Bicubic, pero Pillow es más directo para imágenes.
        )

        if st.button("Escalar Imagen"):
            st.markdown("<h4>Resultados:</h4>")

            original_size = image.size
            new_size = (int(original_size[0] * scale_factor), int(original_size[1] * scale_factor))

            if interpolation_method == "Nearest Neighbor":
                resized_image = image.resize(new_size, Image.NEAREST)
                st.write("**Método:** Vecino Más Cercano")
            elif interpolation_method == "Bilinear":
                resized_image = image.resize(new_size, Image.BILINEAR)
                st.write("**Método:** Bilineal")
            # Agrega más si quieres implementar Bicubic manualmente o con scipy
            # elif interpolation_method == "Bicubic":
            #     resized_image = image.resize(new_size, Image.BICUBIC)
            #     st.write("**Método:** Bicubic")

            st.image(resized_image, caption=f"Imagen Escalada ({interpolation_method})", use_column_width=True)
            st.write(f"Tamaño Original: {original_size[0]}x{original_size[1]} píxeles")
            st.write(f"Tamaño Escalado: {new_size[0]}x{new_size[1]} píxeles")

            st.markdown("""
            <div class="info-block">
            **Nota:** Observa cómo el método de interpolación Vecino Más Cercano puede crear un efecto de "bloqueo" o "pixelado", 
            mientras que la Bilineal produce una imagen más suave al promediar los valores de los píxeles circundantes.
            </div>
            """, unsafe_allow_html=True)

# --- Aplicación 3: Modelado Financiero ---
elif page == "Modelado Financiero":
    st.markdown("<h2 class='subheader'>Aplicación 3: Modelado Financiero (Predicción de Precios de Acciones)</h2>", unsafe_allow_html=True)
    st.markdown("""
    <div class="info-block">
    Si tienes los precios históricos de una acción en fechas específicas, la interpolación te permite estimar el precio 
    en una fecha intermedia para análisis o proyecciones.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<h4>Ingresa tus Datos de Precios de Acciones:</h4>")
    # Datos de ejemplo (pueden ser modificados por el usuario)
    default_dates_str = ["2023-01-01", "2023-01-05", "2023-01-10", "2023-01-15", "2023-01-20"]
    default_prices = [100.0, 102.5, 105.0, 103.0, 106.5]

    num_stock_points = st.number_input("Número de puntos de datos de acciones:", min_value=2, value=len(default_dates_str), step=1, key="num_stock_points")
    
    cols_stock_input = st.columns(2)
    stock_dates = []
    stock_prices = []

    st.write("Ingresa Fecha (YYYY-MM-DD) y Precio de Acción:")
    for i in range(num_stock_points):
        with cols_stock_input[0]:
            d = st.text_input(f"Fecha {i+1} (YYYY-MM-DD):", value=default_dates_str[i] if i < len(default_dates_str) else "2023-01-01", key=f"stock_date_{i}")
            stock_dates.append(d)
        with cols_stock_input[1]:
            p = st.number_input(f"Precio {i+1}:", value=float(default_prices[i]) if i < len(default_prices) else 100.0, key=f"stock_price_{i}")
            stock_prices.append(p)
    
    try:
        # Convertir fechas a números para la interpolación (e.g., marcas de tiempo)
        x_stock_known = np.array([pd.to_datetime(d).timestamp() for d in stock_dates])
        y_stock_known = np.array(stock_prices)
        
        # Normalizar las fechas para el slider si es necesario, o usar directamente timestamps
        min_timestamp = min(x_stock_known)
        max_timestamp = max(x_stock_known)

        st.markdown("<h4>Predice el Precio de la Acción en una Fecha Específica:</h4>")
        predict_date_str = st.text_input("Fecha a Predecir (YYYY-MM-DD):", value="2023-01-12")
        
        if predict_date_str:
            try:
                predict_timestamp = pd.to_datetime(predict_date_str).timestamp()
                if not (min_timestamp <= predict_timestamp <= max_timestamp):
                    st.warning(f"La fecha a predecir debe estar entre {pd.to_datetime(min_timestamp, unit='s').strftime('%Y-%m-%d')} y {pd.to_datetime(max_timestamp, unit='s').strftime('%Y-%m-%d')}.")
                    predict_timestamp = None
            except ValueError:
                st.error("Formato de fecha inválido. Por favor, usa YYYY-MM-DD.")
                predict_timestamp = None
        else:
            predict_timestamp = None

        if st.button("Calcular Predicción de Precios") and predict_timestamp is not None:
            st.markdown("<h4>Resultados:</h4>")

            # Interpolación Lineal
            f_stock_linear = interp1d(x_stock_known, y_stock_known, kind='linear')
            predicted_stock_linear = f_stock_linear(predict_timestamp)
            st.write(f"**Interpolación Lineal:** El precio predicho de la acción el {predict_date_str} es **${predicted_stock_linear:.2f}**")

            # Interpolación Polinomial (Lagrange)
            f_stock_poly = None
            if len(x_stock_known) >= 2:
                try:
                    poly_stock_coeffs = np.polyfit(x_stock_known, y_stock_known, len(x_stock_known) - 1)
                    f_stock_poly = np.poly1d(poly_stock_coeffs)
                    predicted_stock_poly = f_stock_poly(predict_timestamp)
                    st.write(f"**Interpolación Polinomial (Lagrange):** El precio predicho de la acción el {predict_date_str} es **${predicted_stock_poly:.2f}**")
                except np.linalg.LinAlgError:
                    st.warning("No se puede realizar la interpolación polinomial con estos puntos de datos.")
            else:
                st.warning("No hay suficientes puntos para la interpolación polinomial.")

            # Interpolación por Spline Cúbico
            if len(x_stock_known) >= 2:
                f_stock_spline = CubicSpline(x_stock_known, y_stock_known)
                predicted_stock_spline = f_stock_spline(predict_timestamp)
                st.write(f"**Interpolación por Spline Cúbico:** El precio predicho de la acción el {predict_date_str} es **${predicted_stock_spline:.2f}**")
            else:
                st.warning("No hay suficientes puntos para la interpolación por spline cúbico.")

            st.markdown("<h4>Comparación Gráfica:</h4>")
            fig_stock, ax_stock = plt.subplots(figsize=(10, 6))
            
            # Convertir timestamps de vuelta a fechas para el eje X
            dates_plot = [pd.to_datetime(ts, unit='s') for ts in x_stock_known]
            x_interp_stock = np.linspace(min_timestamp, max_timestamp, 500)
            dates_interp_plot = [pd.to_datetime(ts, unit='s') for ts in x_interp_stock]

            ax_stock.plot(dates_plot, y_stock_known, 'o', label='Precios Conocidos', markersize=8, color='red')
            ax_stock.plot(dates_interp_plot, f_stock_linear(x_interp_stock), label='Interpolación Lineal', linestyle='--', color='blue')
            
            if f_stock_poly is not None:
                ax_stock.plot(dates_interp_plot, f_stock_poly(x_interp_stock), label='Interpolación Polinomial', linestyle=':', color='green')
            
            if len(x_stock_known) >= 2:
                ax_stock.plot(dates_interp_plot, f_stock_spline(x_interp_stock), label='Interpolación por Spline Cúbico', linestyle='-.', color='purple')

            ax_stock.axvline(pd.to_datetime(predict_date_str), color='gray', linestyle=':', label=f'Predicción en {predict_date_str}')
            
            ax_stock.set_xlabel("Fecha")
            ax_stock.set_ylabel("Precio de la Acción ($)")
            ax_stock.set_title("Predicción de Precios de Acciones usando Interpolación")
            ax_stock.legend()
            ax_stock.grid(True)
            fig_stock.autofmt_xdate() # Formato automático de fechas
            st.pyplot(fig_stock)

    except ValueError:
        st.error("Asegúrate de que las fechas y los precios sean válidos. Las fechas deben estar en formato YYYY-MM-DD y los precios deben ser números.")
    except Exception as e:
        st.error(f"Ocurrió un error inesperado: {e}")
