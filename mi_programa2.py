# -*- coding: utf-8 -*-
"""
Created on Thu Feb 12 08:57:06 2026

@author: Burga Widinson
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eig

# ==============================================================================
# 1. CONFIGURACION DE LA PAGINA Y ESTILO VISUAL
# ==============================================================================
st.set_page_config(
    page_title="Analisis Sismico Estructural",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Definicion de la Paleta de Colores (Estilo Tecnico Oscuro)
COLOR_FONDO = "#0E1117"     # Fondo principal
COLOR_TEXTO = "#FAFAFA"     # Texto blanco/gris claro
COLOR_ACCENT = "#00ADB5"    # Cian (Elementos principales)
COLOR_SEC = "#393E46"       # Gris (Elementos secundarios)
COLOR_ALERT = "#FF2E63"     # Rojo (Resultados criticos)
COLOR_GRID = "#222831"      # Grilla de fondo

# Inyeccion CSS para forzar el estilo profesional sin emojis
st.markdown(f"""
    <style>
    /* Fondo y texto general */
    .stApp {{
        background-color: {COLOR_FONDO};
        color: {COLOR_TEXTO};
    }}
    /* Personalizacion de metricas (numeros grandes) */
    div[data-testid="stMetricValue"] {{
        color: {COLOR_ACCENT} !important;
        font-family: 'Segoe UI', sans-serif;
        font-weight: 600;
    }}
    /* Encabezados */
    h1, h2, h3 {{
        color: {COLOR_TEXTO} !important;
        font-family: 'Helvetica Neue', sans-serif;
    }}
    /* Botones */
    div.stButton > button:first-child {{
        background-color: {COLOR_ACCENT};
        color: white;
        border-radius: 4px;
        border: none;
        height: 3em;
        font-weight: bold;
        letter-spacing: 1px;
        transition: 0.2s;
    }}
    div.stButton > button:first-child:hover {{
        background-color: #00FFF5;
        color: black;
    }}
    /* Ajustes de graficos */
    figure {{
        border: 1px solid {COLOR_SEC};
        border-radius: 5px;
        padding: 10px;
    }}
    </style>
""", unsafe_allow_html=True)

# ==============================================================================
# 2. FUNCIONES DE CALCULO (MOTOR MATEMATICO)
# ==============================================================================

def interpolar_sitio_asce(Ss, S1, Sitio, Fa_manual, Fv_manual):
    """
    Interpolacion lineal de factores de sitio Fa y Fv segun ASCE 7-16.
    """
    Sitio = Sitio.upper().strip()
    if Sitio in ['E', 'F']: return Fa_manual, Fv_manual
    
    # Puntos de control ASCE 7-16
    Ss_pts = np.array([0.25, 0.50, 0.75, 1.00, 1.25, 1.50])
    S1_pts = np.array([0.10, 0.20, 0.30, 0.40, 0.50, 0.60])
    
    # Datos tabulados simplificados
    tablas = {
        'A': {'Fa': [0.8]*6, 'Fv': [0.8]*6},
        'B': {'Fa': [0.9]*6, 'Fv': [0.8]*6},
        'C': {'Fa': [1.3, 1.3, 1.2, 1.2, 1.2, 1.2], 'Fv': [1.5, 1.5, 1.5, 1.5, 1.5, 1.4]},
        'D': {'Fa': [1.6, 1.4, 1.2, 1.1, 1.0, 1.0], 'Fv': [2.4, 2.2, 2.0, 1.9, 1.8, 1.7]}
    }
    
    if Sitio not in tablas: return 1.0, 1.0
    
    # Interpolacion
    row_Fa = tablas[Sitio]['Fa']
    row_Fv = tablas[Sitio]['Fv']
    
    Ss_c = np.clip(Ss, Ss_pts[0], Ss_pts[-1])
    S1_c = np.clip(S1, S1_pts[0], S1_pts[-1])
    
    Fa_val = np.interp(Ss_c, Ss_pts, row_Fa)
    Fv_val = np.interp(S1_c, S1_pts, row_Fv)
    
    return Fa_val, Fv_val

def ensamblar_matriz_global(K_glob, n1, n2, E, I, A, L, angulo):
    """
    Ensamble de matriz de rigidez local 2D a global.
    """
    k = E * I / (L**3)
    a = E * A / L
    
    # Matriz de rigidez local
    ke = np.array([
        [a, 0, 0, -a, 0, 0],
        [0, 12*k, 6*k*L, 0, -12*k, 6*k*L],
        [0, 6*k*L, 4*k*L**2, 0, -6*k*L, 2*k*L**2],
        [-a, 0, 0, a, 0, 0],
        [0, -12*k, -6*k*L, 0, 12*k, -6*k*L],
        [0, 6*k*L, 2*k*L**2, 0, -6*k*L, 4*k*L**2]
    ])
    
    # Matriz de transformacion
    rad = np.radians(angulo)
    c, s = np.cos(rad), np.sin(rad)
    T = np.zeros((6, 6))
    T[0:3, 0:3] = [[c, s, 0], [-s, c, 0], [0, 0, 1]]
    T[3:6, 3:6] = [[c, s, 0], [-s, c, 0], [0, 0, 1]]
    
    # Rotacion a coordenadas globales
    ke_g = T.T @ ke @ T
    
    # Indices de grados de libertad
    idx = np.array([
        (n1-1)*3, (n1-1)*3+1, (n1-1)*3+2,
        (n2-1)*3, (n2-1)*3+1, (n2-1)*3+2
    ])
    
    # Suma directa a la matriz global
    for i in range(6):
        for j in range(6):
            K_glob[idx[i], idx[j]] += ke_g[i, j]
            
    return K_glob

# ==============================================================================
# 3. INTERFAZ DE USUARIO (PANEL LATERAL E INPUTS)
# ==============================================================================

with st.sidebar:
    st.header("Configuracion del Modelo")
    st.markdown("---")
    
    # Bloque 1: Sismicidad
    with st.expander("1. Parametros Sismicos", expanded=True):
        Ss = st.number_input("Ss (Aceleracion Corta)", value=1.2, format="%.2f")
        S1 = st.number_input("S1 (Aceleracion 1s)", value=1.0, format="%.2f")
        Sitio = st.selectbox("Clase de Sitio", ['A', 'B', 'C', 'D', 'E', 'F'], index=3)
        TL = st.number_input("TL (Periodo Largo)", value=8.0)

    # Bloque 2: Geometria
    with st.expander("2. Geometria del Portico", expanded=False):
        nP = st.number_input("Numero de Pisos", value=6, min_value=1)
        nV = st.number_input("Numero de Vanos", value=2, min_value=1)
        H_piso = st.number_input("Altura de Entrepiso (m)", value=3.5)
        L_vano = st.number_input("Longitud de Vano (m)", value=5.0)

    # Bloque 3: Materiales
    with st.expander("3. Propiedades y Secciones", expanded=False):
        E_mod = st.number_input("Modulo Elasticidad E (Pa)", value=2.15e11, format="%.2e")
        m_nudo = st.number_input("Masa por Nudo (kg)", value=20000.0)
        zeta = st.slider("Amortiguamiento", 0.01, 0.10, 0.05)
        
        st.markdown("**Dimensiones de Seccion (m)**")
        c1, c2 = st.columns(2)
        bc, hc = c1.number_input("Base Col", 0.45), c2.number_input("Altura Col", 0.45)
        v1, v2 = st.columns(2)
        bv, hv = v1.number_input("Base Viga", 0.30), v2.number_input("Altura Viga", 0.40)

    st.markdown("---")
    btn_calc = st.button("CALCULAR RESULTADOS", type="primary", use_container_width=True)

# ==============================================================================
# 4. LOGICA PRINCIPAL (EJECUCION AL PRESIONAR BOTON)
# ==============================================================================

# Titulo principal de la aplicacion
st.title("Plataforma de Analisis Estructural")
st.markdown("Herramienta avanzada para analisis modal espectral segun norma ASCE 7-16.")

if btn_calc:
    with st.spinner('Procesando matrices de rigidez y resolviendo autovalores...'):
        
        # --- A. PREPROCESAMIENTO ---
        # Calculo de propiedades geometricas
        Ic, Ac = (bc * hc**3) / 12, bc * hc
        Iv, Av = (bv * hv**3) / 12, bv * hv
        
        # Definicion de la malla
        nx_nudos = nV + 1
        ny_nudos = nP + 1
        GDL_totales = (nx_nudos * ny_nudos) * 3
        
        # Inicializacion de matrices
        K_sistema = np.zeros((GDL_totales, GDL_totales))
        M_sistema = np.zeros((GDL_totales, GDL_totales))
        
        # Listas para guardar coordenadas (visualizacion)
        plot_vigas = []
        plot_columnas = []
        plot_masas = []
        count_masas = 0
        
        # --- B. ENSAMBLE MATRICIAL ---
        for j in range(1, ny_nudos + 1):
            for i in range(1, nx_nudos + 1):
                nudo_actual = (j - 1) * nx_nudos + i
                x_pos = (i - 1) * L_vano
                y_pos = (j - 1) * H_piso
                
                # Asignacion de Masa (Solo en pisos elevados)
                if j > 1:
                    idx_m = (nudo_actual - 1) * 3
                    M_sistema[idx_m, idx_m] = m_nudo     # Masa en X
                    M_sistema[idx_m+1, idx_m+1] = m_nudo # Masa en Y
                    plot_masas.append((x_pos, y_pos))
                    count_masas += 1
                
                # Elemento Columna (Conecta con el nudo superior)
                if j < ny_nudos:
                    nudo_sup = nudo_actual + nx_nudos
                    K_sistema = ensamblar_matriz_global(K_sistema, nudo_actual, nudo_sup, E_mod, Ic, Ac, H_piso, 90)
                    plot_columnas.append([(x_pos, y_pos), (x_pos, y_pos + H_piso)])
                
                # Elemento Viga (Conecta con el nudo derecho)
                if i < nx_nudos and j > 1:
                    nudo_der = nudo_actual + 1
                    K_sistema = ensamblar_matriz_global(K_sistema, nudo_actual, nudo_der, E_mod, Iv, Av, L_vano, 0)
                    plot_vigas.append([(x_pos, y_pos), (x_pos + L_vano, y_pos)])

        # --- C. SOLUCION NUMERICA (EIGENVALUES) ---
        # Identificar grados de libertad libres (asumiendo empotramiento perfecto en la base)
        gdl_base = nx_nudos * 3
        gdl_libres = np.arange(gdl_base, GDL_totales)
        
        # Reduccion de matrices
        K_red = K_sistema[np.ix_(gdl_libres, gdl_libres)]
        M_red = M_sistema[np.ix_(gdl_libres, gdl_libres)]
        
        # Solucion del problema de valores propios usando SCIPY (Mas estable que Numpy en Cloud)
        vals_propios, vecs_propios = eig(K_red, M_red)
        
        # Procesamiento de resultados
        omegas_sq = np.sort(np.real(vals_propios)) # Parte real ordenada
        omegas_sq = omegas_sq[omegas_sq > 1e-5]    # Filtrar valores nulos/negativos
        
        periodos = 2 * np.pi / np.sqrt(omegas_sq)
        T1 = periodos[0]
        T2 = periodos[1] # Periodo del segundo modo
        
        # Coeficientes Rayleigh
        w1, w2 = 2*np.pi/T1, 2*np.pi/T2
        try:
            rayleigh = np.linalg.solve([[1/(2*w1), w1/2], [1/(2*w2), w2/2]], [zeta, zeta])
            alpha_R, beta_R = rayleigh[0], rayleigh[1]
        except:
            alpha_R, beta_R = 0.0, 0.0

        # --- D. CALCULO ESPECTRAL (ASCE 7-16) ---
        # Factores de sitio interpolados
        Fa, Fv = interpolar_sitio_asce(Ss, S1, Sitio, 1.0, 1.0)
        
        # Parametros espectrales
        SMS, SM1 = Fa * Ss, Fv * S1
        SDS, SD1 = (2/3) * SMS, (2/3) * SM1
        Ts = SD1 / SDS if SDS > 0 else 0
        T0 = 0.2 * Ts
        
        # Aceleracion de diseno (Sa) para T1
        T_target = T1
        if T_target <= T0: Sa_d = SDS * (0.4 + 0.6 * T_target / max(T0, 1e-4))
        elif T_target <= Ts: Sa_d = SDS
        elif T_target <= TL: Sa_d = SD1 / T_target
        else: Sa_d = (SD1 * TL) / (T_target**2)
        
        # Calculo de fuerzas (W y V)
        peso_total_ton = (count_masas * m_nudo) / 1000 # Peso "sismico" (toneladas masa ref)
        cortante_basal = Sa_d * peso_total_ton * 9.81  # V = m * Sa (kN) si usamos SI, aqui ajustado a TonF aprox segun logica usuario
        # Ajustando a la logica previa del usuario: V (Ton) = Sa(g) * W(Ton)
        cortante_basal_usuario = Sa_d * peso_total_ton

        # ==============================================================================
        # 5. VISUALIZACION DE RESULTADOS
        # ==============================================================================
        
        tab_res, tab_det = st.tabs(["Resultados Graficos", "Detalles Tecnicos"])
        
        with tab_res:
            # Metricas Principales
            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
            col_m1.metric("Periodo T1", f"{T1:.3f} s")
            col_m2.metric("Periodo T2", f"{T2:.3f} s") # Nuevo requerimiento
            col_m3.metric("Cortante Basal", f"{cortante_basal_usuario:.2f} Ton")
            col_m4.metric("Aceleracion Sa", f"{Sa_d:.3f} g")
            
            st.markdown("---")
            
            col_g1, col_g2 = st.columns([1, 1])
            
            # Grafico 1: Modelo Estructural
            with col_g1:
                st.subheader("Modelo Discretizado")
                fig1, ax1 = plt.subplots(figsize=(5, 6))
                fig1.patch.set_facecolor(COLOR_FONDO)
                ax1.set_facecolor(COLOR_FONDO)
                
                # Dibujo
                for linea in plot_vigas:
                    ax1.plot([p[0] for p in linea], [p[1] for p in linea], color=COLOR_ACCENT, lw=2.5)
                for linea in plot_columnas:
                    ax1.plot([p[0] for p in linea], [p[1] for p in linea], color=COLOR_SEC, lw=3.5)
                
                xs = [p[0] for p in plot_masas]
                ys = [p[1] for p in plot_masas]
                ax1.scatter(xs, ys, color=COLOR_ALERT, s=80, zorder=5, label="Nudos Masa")
                
                # Estilo
                ax1.set_xlabel("Distancia (m)", color=COLOR_TEXTO)
                ax1.set_ylabel("Altura (m)", color=COLOR_TEXTO)
                ax1.tick_params(colors=COLOR_TEXTO)
                ax1.grid(True, linestyle=':', color=COLOR_GRID)
                ax1.spines['top'].set_visible(False)
                ax1.spines['right'].set_visible(False)
                ax1.spines['bottom'].set_color(COLOR_SEC)
                ax1.spines['left'].set_color(COLOR_SEC)
                ax1.set_aspect('equal')
                st.pyplot(fig1)

            # Grafico 2: Espectro de Diseno
            with col_g2:
                st.subheader("Espectro de Respuesta")
                fig2, ax2 = plt.subplots(figsize=(6, 5))
                fig2.patch.set_facecolor(COLOR_FONDO)
                ax2.set_facecolor(COLOR_FONDO)
                
                # Generacion curva espectral
                T_plot = np.linspace(0.01, 5.0, 200)
                Sa_plot = []
                for t in T_plot:
                    if t <= T0: val = SDS * (0.4 + 0.6 * t / max(T0, 1e-4))
                    elif t <= Ts: val = SDS
                    elif t <= TL: val = SD1 / t
                    else: val = (SD1 * TL) / (t**2)
                    Sa_plot.append(val)
                
                ax2.plot(T_plot, Sa_plot, color=COLOR_TEXTO, lw=2, label="ASCE 7-16")
                
                # Marcadores
                ax2.plot([T1, T1], [0, Sa_d], linestyle='--', color=COLOR_ACCENT)
                ax2.scatter(T1, Sa_d, color=COLOR_ALERT, s=100, zorder=10, label=f"T1 ({T1:.2f}s)")
                
                # Marcador T2
                if T2 <= T0: Sa_d2 = SDS * (0.4 + 0.6 * T2 / max(T0, 1e-4))
                elif T2 <= Ts: Sa_d2 = SDS
                else: Sa_d2 = SD1 / T2
                ax2.scatter(T2, Sa_d2, color=COLOR_ACCENT, marker='^', s=80, zorder=10, label=f"T2 ({T2:.2f}s)")

                # Estilo
                ax2.set_xlabel("Periodo (s)", color=COLOR_TEXTO)
                ax2.set_ylabel("Aceleracion Espectral (g)", color=COLOR_TEXTO)
                ax2.tick_params(colors=COLOR_TEXTO)
                ax2.grid(True, linestyle=':', color=COLOR_GRID)
                ax2.legend(facecolor=COLOR_FONDO, labelcolor=COLOR_TEXTO, edgecolor=COLOR_SEC)
                ax2.spines['top'].set_visible(False)
                ax2.spines['right'].set_visible(False)
                ax2.spines['bottom'].set_color(COLOR_SEC)
                ax2.spines['left'].set_color(COLOR_SEC)
                st.pyplot(fig2)

        with tab_det:
            st.markdown("#### Memoria de Calculo Interna")
            st.info("Los siguientes parametros han sido calculados automaticamente:")
            
            c_d1, c_d2 = st.columns(2)
            with c_d1:
                st.write("**Factores de Sitio:**")
                st.write(f"- Fa: {Fa:.4f}")
                st.write(f"- Fv: {Fv:.4f}")
                st.write("**Parametros Espectrales:**")
                st.write(f"- SDS: {SDS:.4f} g")
                st.write(f"- SD1: {SD1:.4f} g")
            
            with c_d2:
                st.write("**Propiedades Dinamicas:**")
                st.write(f"- Masa Participante: {peso_total_ton:.2f} Ton")
                st.write(f"- Coef. Rayleigh Alpha: {alpha_R:.5f}")
                st.write(f"- Coef. Rayleigh Beta: {beta_R:.5f}")

else:
    st.info("Sistema en espera. Configure los parametros en el panel lateral y presione el boton de calculo.")