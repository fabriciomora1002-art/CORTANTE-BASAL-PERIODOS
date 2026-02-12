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
# 1. CONFIGURACION DE LA PAGINA Y ESTILO "CLEAN / LIGHT"
# ==============================================================================
st.set_page_config(
    page_title="Cálculo de Coeficientes de Rayleigh",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Paleta de Colores (Estilo Ingenieria Civil - Claro/Plomo)
COLOR_FONDO_APP = "#F0F2F6"      # Plomo muy suave
COLOR_CARD = "rgba(255, 255, 255, 0.95)" # Blanco cristal
COLOR_TEXTO_PRINCIPAL = "#262730" # Gris oscuro
COLOR_ACCENT = "#007BFF"         # Azul Ingenieria
COLOR_COLUMNA = "#4B5563"        # Gris Acero
COLOR_ALERT = "#DC3545"          # Rojo suave

st.markdown(f"""
    <style>
    /* Estilos Generales */
    .stApp {{
        background-color: {COLOR_FONDO_APP};
        color: {COLOR_TEXTO_PRINCIPAL};
    }}
    section[data-testid="stSidebar"] {{
        background-color: #E5E7EB;
        color: {COLOR_TEXTO_PRINCIPAL};
    }}
    .stMarkdown, .stText, h1, h2, h3, h4, h5, h6, label {{
        color: {COLOR_TEXTO_PRINCIPAL} !important;
        font-family: 'Segoe UI', Helvetica, sans-serif;
    }}
    
    /* Tarjetas de Metricas */
    div[data-testid="stMetric"] {{
        background-color: {COLOR_CARD};
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        border: 1px solid #D1D5DB;
    }}
    div[data-testid="stMetricValue"] {{
        color: {COLOR_ACCENT} !important;
        font-weight: 700;
    }}
    div[data-testid="stMetricLabel"] {{
        color: #6B7280 !important;
    }}

    /* Boton Principal */
    div.stButton > button:first-child {{
        background-color: {COLOR_ACCENT};
        color: white;
        border: none;
        border-radius: 6px;
        font-weight: 600;
        box-shadow: 0 2px 5px rgba(0,123,255,0.3);
    }}
    div.stButton > button:first-child:hover {{
        background-color: #0056b3;
        color: white;
    }}
    
    /* Graficos */
    figure {{
        border: 1px solid #E5E7EB;
        border-radius: 8px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.04);
        background-color: white;
    }}
    </style>
""", unsafe_allow_html=True)

# ==============================================================================
# 2. FUNCIONES DE CALCULO
# ==============================================================================

def interpolar_sitio_asce(Ss, S1, Sitio, Fa_manual, Fv_manual):
    """Interpolacion lineal ASCE 7-16."""
    Sitio = Sitio.upper().strip()
    if Sitio in ['E', 'F']: return Fa_manual, Fv_manual
    
    Ss_pts = np.array([0.25, 0.50, 0.75, 1.00, 1.25, 1.50])
    S1_pts = np.array([0.10, 0.20, 0.30, 0.40, 0.50, 0.60])
    
    tablas = {
        'A': {'Fa': [0.8]*6, 'Fv': [0.8]*6},
        'B': {'Fa': [0.9]*6, 'Fv': [0.8]*6},
        'C': {'Fa': [1.3, 1.3, 1.2, 1.2, 1.2, 1.2], 'Fv': [1.5, 1.5, 1.5, 1.5, 1.5, 1.4]},
        'D': {'Fa': [1.6, 1.4, 1.2, 1.1, 1.0, 1.0], 'Fv': [2.4, 2.2, 2.0, 1.9, 1.8, 1.7]}
    }
    
    if Sitio not in tablas: return 1.0, 1.0
    
    Fa_val = np.interp(np.clip(Ss, Ss_pts[0], Ss_pts[-1]), Ss_pts, tablas[Sitio]['Fa'])
    Fv_val = np.interp(np.clip(S1, S1_pts[0], S1_pts[-1]), S1_pts, tablas[Sitio]['Fv'])
    
    return Fa_val, Fv_val

def ensamblar_matriz_global(K_glob, n1, n2, E, I, A, L, angulo):
    """Ensamble de rigidez 2D."""
    k = E * I / (L**3)
    a = E * A / L
    ke = np.array([
        [a, 0, 0, -a, 0, 0],
        [0, 12*k, 6*k*L, 0, -12*k, 6*k*L],
        [0, 6*k*L, 4*k*L**2, 0, -6*k*L, 2*k*L**2],
        [-a, 0, 0, a, 0, 0],
        [0, -12*k, -6*k*L, 0, 12*k, -6*k*L],
        [0, 6*k*L, 2*k*L**2, 0, -6*k*L, 4*k*L**2]
    ])
    rad = np.radians(angulo)
    c, s = np.cos(rad), np.sin(rad)
    T = np.zeros((6, 6))
    T[0:3, 0:3] = [[c, s, 0], [-s, c, 0], [0, 0, 1]]
    T[3:6, 3:6] = [[c, s, 0], [-s, c, 0], [0, 0, 1]]
    ke_g = T.T @ ke @ T
    idx = np.array([(n1-1)*3, (n1-1)*3+1, (n1-1)*3+2, (n2-1)*3, (n2-1)*3+1, (n2-1)*3+2])
    for i in range(6):
        for j in range(6): K_glob[idx[i], idx[j]] += ke_g[i, j]
    return K_glob

# ==============================================================================
# 3. INTERFAZ DE USUARIO (PANEL LATERAL)
# ==============================================================================

with st.sidebar:
    st.header("Configuracion del Modelo")
    
    with st.expander("1. Parametros Sismicos", expanded=True):
        Ss = st.number_input("Ss (Aceleracion Corta)", value=1.2, format="%.2f")
        S1 = st.number_input("S1 (Aceleracion 1s)", value=1.0, format="%.2f")
        Sitio = st.selectbox("Clase de Sitio", ['A', 'B', 'C', 'D', 'E', 'F'], index=3)
        TL = st.number_input("TL (Periodo Largo)", value=8.0)

    with st.expander("2. Geometria del Portico", expanded=False):
        nP = st.number_input("Numero de Pisos", value=6, min_value=1)
        nV = st.number_input("Numero de Vanos", value=2, min_value=1)
        H_piso = st.number_input("Altura de Entrepiso (m)", value=3.5)
        L_vano = st.number_input("Longitud de Vano (m)", value=5.0)

    with st.expander("3. Propiedades y Secciones", expanded=False):
        E_mod = st.number_input("Modulo Elasticidad E (Pa)", value=2.15e11, format="%.2e")
        m_nudo = st.number_input("Masa por Nudo (kg)", value=20000.0)
        zeta = st.slider("Amortiguamiento (Zeta)", 0.01, 0.10, 0.05)
        
        st.write("**Dimensiones (m)**")
        c1, c2 = st.columns(2)
        bc, hc = c1.number_input("Base Col", 0.45), c2.number_input("Altura Col", 0.45)
        v1, v2 = st.columns(2)
        bv, hv = v1.number_input("Base Viga", 0.30), v2.number_input("Altura Viga", 0.40)

    st.markdown("---")
    btn_calc = st.button("CALCULAR RESULTADOS", type="primary", use_container_width=True)

# ==============================================================================
# 4. LOGICA PRINCIPAL
# ==============================================================================

st.title("Cálculo de Coeficientes de Rayleigh")
st.write("Cálculo de cortante basal usando ASCE 7-16")

if btn_calc:
    with st.spinner('Procesando matriz de rigidez y solucionando eigenvalores...'):
        
        # --- A. CALCULO MATRICIAL ---
        Ic, Ac = (bc * hc**3) / 12, bc * hc
        Iv, Av = (bv * hv**3) / 12, bv * hv
        nx, ny = nV + 1, nP + 1
        GDL = (nx * ny) * 3
        K_sis, M_sis = np.zeros((GDL, GDL)), np.zeros((GDL, GDL))
        
        plot_vigas, plot_cols, plot_masas = [], [], []
        count_masas = 0
        
        for j in range(1, ny + 1):
            for i in range(1, nx + 1):
                nudo = (j - 1) * nx + i
                x, y = (i - 1) * L_vano, (j - 1) * H_piso
                
                if j > 1: # Masas
                    idx_m = (nudo - 1) * 3
                    M_sis[idx_m, idx_m] = m_nudo
                    M_sis[idx_m+1, idx_m+1] = m_nudo
                    plot_masas.append((x, y))
                    count_masas += 1
                
                if j < ny: # Columnas
                    K_sis = ensamblar_matriz_global(K_sis, nudo, nudo + nx, E_mod, Ic, Ac, H_piso, 90)
                    plot_cols.append([(x, y), (x, y + H_piso)])
                
                if i < nx and j > 1: # Vigas
                    K_sis = ensamblar_matriz_global(K_sis, nudo, nudo + 1, E_mod, Iv, Av, L_vano, 0)
                    plot_vigas.append([(x, y), (x + L_vano, y)])

        # --- B. SOLVER EIGENVALUES ---
        gdl_libres = np.arange(nx * 3, GDL)
        K_red = K_sis[np.ix_(gdl_libres, gdl_libres)]
        M_red = M_sis[np.ix_(gdl_libres, gdl_libres)]
        
        # Solver Scipy (Robusto)
        vals, vecs = eig(K_red, M_red)
        w2_sorted = np.sort(np.real(vals))
        w2_valid = w2_sorted[w2_sorted > 1e-5] # Filtro ruido numerico
        
        T1 = 2 * np.pi / np.sqrt(w2_valid[0])
        T2 = 2 * np.pi / np.sqrt(w2_valid[1])
        
        # --- C. CALCULO DE RAYLEIGH (NUEVO) ---
        w1 = 2 * np.pi / T1
        w2 = 2 * np.pi / T2
        
        # Sistema de ecuaciones: 
        # 0.5/w * alpha + 0.5*w * beta = zeta (para w1 y w2)
        A_ray = np.array([
            [1/(2*w1), w1/2], 
            [1/(2*w2), w2/2]
        ])
        b_ray = np.array([zeta, zeta])
        
        try:
            coeffs_ray = np.linalg.solve(A_ray, b_ray)
            alpha_R, beta_R = coeffs_ray[0], coeffs_ray[1]
        except:
            alpha_R, beta_R = 0.0, 0.0

        # --- D. CALCULO ESPECTRAL ---
        Fa, Fv = interpolar_sitio_asce(Ss, S1, Sitio, 1.0, 1.0)
        SMS, SM1 = Fa * Ss, Fv * S1
        SDS, SD1 = (2/3) * SMS, (2/3) * SM1
        Ts = SD1 / SDS if SDS > 0 else 0
        T0 = 0.2 * Ts
        
        if T1 <= T0: Sa_d = SDS * (0.4 + 0.6 * T1 / max(T0, 1e-4))
        elif T1 <= Ts: Sa_d = SDS
        elif T1 <= TL: Sa_d = SD1 / T1
        else: Sa_d = (SD1 * TL) / (T1**2)
        
        peso_total = (count_masas * m_nudo) / 1000
        cortante = Sa_d * peso_total

        # ==============================================================================
        # 5. VISUALIZACION
        # ==============================================================================
        
        tab_main, tab_data = st.tabs(["Panel de Resultados", "Datos Tecnicos"])
        
        with tab_main:
            # Metricas
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Periodo Fundamental T1", f"{T1:.3f} s")
            c2.metric("Periodo Modo 2 (T2)", f"{T2:.3f} s")
            c3.metric("Cortante Basal", f"{cortante:.2f} Ton")
            c4.metric("Aceleracion Sa", f"{Sa_d:.3f} g")
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            col_g1, col_g2 = st.columns(2)
            
            # Grafico Estructura
            with col_g1:
                st.subheader("Discretizacion del Modelo")
                fig1, ax1 = plt.subplots(figsize=(6, 5))
                fig1.patch.set_facecolor('white')
                ax1.set_facecolor('white')
                
                for l in plot_vigas: ax1.plot([p[0] for p in l], [p[1] for p in l], color=COLOR_ACCENT, lw=3)
                for l in plot_cols: ax1.plot([p[0] for p in l], [p[1] for p in l], color=COLOR_COLUMNA, lw=4)
                xs, ys = [p[0] for p in plot_masas], [p[1] for p in plot_masas]
                ax1.scatter(xs, ys, color=COLOR_ALERT, s=100, zorder=5, edgecolors='black', label="Masa")
                
                ax1.set_xlabel("Longitud (m)", color='black')
                ax1.set_ylabel("Altura (m)", color='black')
                ax1.tick_params(colors='black')
                ax1.grid(True, linestyle=':', color='#E5E7EB')
                ax1.set_aspect('equal')
                st.pyplot(fig1)

            # Grafico Espectro
            with col_g2:
                st.subheader("Espectro ASCE 7-16")
                fig2, ax2 = plt.subplots(figsize=(6, 5))
                fig2.patch.set_facecolor('white')
                ax2.set_facecolor('white')
                
                T_plot = np.linspace(0.01, 5.0, 200)
                Sa_plot = []
                for t in T_plot:
                    if t <= T0: v = SDS*(0.4+0.6*t/max(T0, 1e-4))
                    elif t <= Ts: v = SDS
                    elif t <= TL: v = SD1/t
                    else: v = (SD1*TL)/(t**2)
                    Sa_plot.append(v)
                
                ax2.plot(T_plot, Sa_plot, color='#374151', lw=2)
                ax2.plot([T1, T1], [0, Sa_d], '--', color=COLOR_ALERT)
                ax2.scatter(T1, Sa_d, color=COLOR_ALERT, s=120, label=f"T1 ({T1:.2f}s)")
                
                if T2 <= Ts: Sa_2 = SDS
                else: Sa_2 = SD1/T2
                ax2.scatter(T2, Sa_2, color=COLOR_ACCENT, marker='^', s=100, label=f"T2 ({T2:.2f}s)")
                
                ax2.set_xlabel("Periodo (s)", color='black')
                ax2.set_ylabel("Sa (g)", color='black')
                ax2.tick_params(colors='black')
                ax2.grid(True, linestyle=':', color='#E5E7EB')
                st.pyplot(fig2)

        with tab_data:
            st.markdown("### Coeficientes de Amortiguamiento (Rayleigh)")
            st.markdown("Calculados resolviendo el sistema para $T_1$ y $T_2$ con el amortiguamiento objetivo:")
            
            # Mostramos los coeficientes en tarjetas grandes dentro de esta pestaña
            r1, r2, r3 = st.columns(3)
            r1.metric("Coeficiente Alpha", f"{alpha_R:.5f}")
            r2.metric("Coeficiente Beta", f"{beta_R:.5f}")
            r3.metric("Zeta Objetivo", f"{zeta:.2f}")

            st.markdown("---")
            st.info("Desglose de parametros internos:")
            
            c_d1, c_d2 = st.columns(2)
            c_d1.write(f"**Factores Sitio:** Fa={Fa:.3f}, Fv={Fv:.3f}")
            c_d1.write(f"**Parametros:** SDS={SDS:.3f}, SD1={SD1:.3f}")
            c_d2.write(f"**Masa Total Participante:** {peso_total:.2f} Ton")

else:
    st.info("Configure los parametros en el panel lateral y presione CALCULAR RESULTADOS.")