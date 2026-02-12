# -*- coding: utf-8 -*-
"""
Created on Thu Feb 12 08:57:06 2026

@author: Usuario
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eig
import matplotlib.patches as patches

# ==========================================
# 1. CONFIGURACIÓN DE LA PÁGINA Y ESTILO
# ==========================================
st.set_page_config(
    page_title="SeismoStruct Pro | Análisis Espectral",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Definición de Paleta de Colores (Estilo "Ingeniería High-Tech")
# Extraída conceptualmente para dar un look profesional y ganador
COLORS = {
    "background": "#0E1117",      # Fondo oscuro profesional
    "text": "#FAFAFA",            # Texto claro
    "accent": "#00A8E8",          # Azul Cián (Elementos estructurales)
    "highlight": "#FF9F1C",       # Naranja (Resultados críticos / Alertas)
    "secondary": "#4B5563",       # Gris medio (Ejes y guías)
    "success": "#2ECC71",         # Verde (Verificaciones)
    "grid": "#333333"             # Grilla sutil
}

# Inyectamos CSS personalizado para mejorar la interfaz (botones, métricas, tablas)
st.markdown(f"""
    <style>
    /* Estilo de las métricas */
    div[data-testid="stMetricValue"] {{
        font-size: 1.5rem !important;
        color: {COLORS['highlight']} !important;
        font-family: 'Roboto Mono', monospace;
    }}
    /* Estilo de los headers */
    h1, h2, h3 {{
        font-family: 'Helvetica Neue', sans-serif;
        font-weight: 700;
        letter-spacing: -0.5px;
    }}
    /* Bordes redondeados en los contenedores */
    .stCard {{
        border-radius: 10px;
        border: 1px solid {COLORS['secondary']};
    }}
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. LÓGICA DE CÁLCULO (BACKEND)
# ==========================================

class StructuralSolver:
    """Clase encargada de todos los cálculos matriciales y espectrales."""
    
    @staticmethod
    def interpolar_sitio(Ss, S1, clase_sitio, Fa_man, Fv_man):
        """Interpolación lineal para coeficientes de sitio ASCE 7-16."""
        if clase_sitio in ['E', 'F']: return Fa_man, Fv_man
        
        # Tablas simplificadas ASCE 7-16 (Solo valores referenciales para el ejemplo)
        # Nota: En un software real, estas tablas deben ser exactas a la norma.
        tables = {
            'A': {'Fa': [0.8]*6, 'Fv': [0.8]*6},
            'B': {'Fa': [0.9]*6, 'Fv': [0.8]*6},
            'C': {'Fa': [1.3, 1.3, 1.2, 1.2, 1.2, 1.2], 'Fv': [1.5, 1.5, 1.5, 1.5, 1.5, 1.4]},
            'D': {'Fa': [1.6, 1.4, 1.2, 1.1, 1.0, 1.0], 'Fv': [2.4, 2.2, 2.0, 1.9, 1.8, 1.7]}
        }
        
        row_Fa = tables.get(clase_sitio, tables['D'])['Fa']
        row_Fv = tables.get(clase_sitio, tables['D'])['Fv']
        
        Ss_pts = [0.25, 0.50, 0.75, 1.00, 1.25, 1.50]
        S1_pts = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60]
        
        Fa = np.interp(min(max(Ss, 0.25), 1.5), Ss_pts, row_Fa)
        Fv = np.interp(min(max(S1, 0.1), 0.6), S1_pts, row_Fv)
        return Fa, Fv

    @staticmethod
    def stiffness_matrix_2d(E, I, A, L):
        """Matriz de rigidez local para un elemento pórtico 2D."""
        k = E * I / (L**3)
        a = E * A / L
        return np.array([
            [a, 0, 0, -a, 0, 0],
            [0, 12*k, 6*k*L, 0, -12*k, 6*k*L],
            [0, 6*k*L, 4*k*L**2, 0, -6*k*L, 2*k*L**2],
            [-a, 0, 0, a, 0, 0],
            [0, -12*k, -6*k*L, 0, 12*k, -6*k*L],
            [0, 6*k*L, 2*k*L**2, 0, -6*k*L, 4*k*L**2]
        ])

    @staticmethod
    def transformation_matrix(angle_deg):
        """Matriz de transformación de coordenadas."""
        rad = np.radians(angle_deg)
        c, s = np.cos(rad), np.sin(rad)
        T = np.zeros((6, 6))
        T[0:3, 0:3] = [[c, s, 0], [-s, c, 0], [0, 0, 1]]
        T[3:6, 3:6] = [[c, s, 0], [-s, c, 0], [0, 0, 1]]
        return T

    @staticmethod
    def assemble_global(K_global, node_i, node_j, E, I, A, L, angle_deg):
        """Ensambla la matriz local en la global."""
        Ke_local = StructuralSolver.stiffness_matrix_2d(E, I, A, L)
        T = StructuralSolver.transformation_matrix(angle_deg)
        Ke_global = T.T @ Ke_local @ T
        
        # Índices de grados de libertad (3 por nudo)
        idx = np.r_[ (node_i-1)*3 : (node_i-1)*3+3, (node_j-1)*3 : (node_j-1)*3+3 ]
        
        for i in range(6):
            for j in range(6):
                K_global[idx[i], idx[j]] += Ke_global[i, j]
        return K_global

# ==========================================
# 3. INTERFAZ DE USUARIO (FRONTEND)
# ==========================================

# --- BARRA LATERAL (Inputs ordenados) ---
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/structural.png", width=60) # Logo placeholder
    st.title("Configuración")
    
    with st.expander(" 1. Parámetros Sísmicos", expanded=True):
        ss_input = st.number_input("$S_s$ (Aceleración Corta)", 0.0, 3.0, 1.2, 0.1, help="Aceleración espectral para periodo corto")
        s1_input = st.number_input("$S_1$ (Aceleración 1s)", 0.0, 2.0, 0.6, 0.1)
        tl_input = st.number_input("$T_L$ (Periodo Largo)", 4.0, 12.0, 8.0, 1.0)
        site_class = st.selectbox("Clase de Sitio", ["A", "B", "C", "D", "E"], index=3)

    with st.expander(" 2. Geometría del Pórtico", expanded=True):
        col_g1, col_g2 = st.columns(2)
        n_stories = col_g1.number_input("N° Pisos", 1, 50, 6)
        n_bays = col_g2.number_input("N° Vanos", 1, 10, 2)
        h_story = st.slider("Altura de Entrepiso (m)", 2.5, 6.0, 3.5, 0.1)
        l_bay = st.slider("Longitud de Vano (m)", 3.0, 10.0, 5.0, 0.5)

    with st.expander(" 3. Materiales y Secciones", expanded=False):
        E_mod = st.number_input("Módulo Elasticidad $E$ (Pa)", value=2.15e10, format="%.2e")
        mass_node = st.number_input("Masa Sísmica por Nudo (kg)", value=20000.0, step=1000.0)
        col_m1, col_m2 = st.columns(2)
        b_col, h_col = col_m1.number_input("b Col (m)", 0.3), col_m2.number_input("h Col (m)", 0.5)
        b_beam, h_beam = col_m1.number_input("b Viga (m)", 0.3), col_m2.number_input("h Viga (m)", 0.4)
        zeta = st.slider("Amortiguamiento $\zeta$", 0.02, 0.10, 0.05, 0.01)

    btn_calc = st.button(" EJECUTAR ANÁLISIS", type="primary", use_container_width=True)
    st.caption("v2.0 | Concurso de Ingeniería")

# --- ÁREA PRINCIPAL ---
st.title(" Analizador Sísmico Avanzado")
st.markdown("Plataforma de análisis modal espectral para pórticos planos según **ASCE 7-16**.")

if btn_calc:
    with st.spinner("Ensamblando matrices y resolviendo valores propios..."):
        
        # --- PRE-CÁLCULO DE PROPIEDADES ---
        # Inercias y Áreas
        Ic = (b_col * h_col**3) / 12
        Ac = b_col * h_col
        Iv = (b_beam * h_beam**3) / 12
        Av = b_beam * h_beam
        
        # Geometría de Nudos
        nx_nodes = n_bays + 1
        ny_nodes = n_stories + 1
        total_dof = (nx_nodes * ny_nodes) * 3
        
        # Inicialización de Matrices
        K_glob = np.zeros((total_dof, total_dof))
        M_glob = np.zeros((total_dof, total_dof))
        
        # Listas para guardar coordenadas (para el gráfico)
        plot_beams = []
        plot_cols = []
        node_coords = []
        mass_nodes_count = 0

        # --- ENSAMBLE MATRICIAL ---
        for j in range(1, ny_nodes + 1): # Loop Pisos (Y)
            for i in range(1, nx_nodes + 1): # Loop Vanos (X)
                current_node = (j - 1) * nx_nodes + i
                x = (i - 1) * l_bay
                y = (j - 1) * h_story
                node_coords.append((x, y))

                # Asignación de Masa (Solo en pisos, no en base)
                if j > 1:
                    idx_m = (current_node - 1) * 3
                    # Masa traslacional X e Y
                    M_glob[idx_m, idx_m] = mass_node
                    M_glob[idx_m+1, idx_m+1] = mass_node
                    mass_nodes_count += 1
                
                # Elemento Columna (hacia abajo, excepto piso 1 que conecta a 0 virtualmente o piso j a j+1)
                # Lógica: Conectamos nudo actual con el de arriba
                if j < ny_nodes:
                    node_top = current_node + nx_nodes
                    K_glob = StructuralSolver.assemble_global(K_glob, current_node, node_top, E_mod, Ic, Ac, h_story, 90)
                    plot_cols.append([(x, y), (x, y + h_story)])
                
                # Elemento Viga (hacia la derecha)
                if i < nx_nodes and j > 1:
                    node_right = current_node + 1
                    K_glob = StructuralSolver.assemble_global(K_glob, current_node, node_right, E_mod, Iv, Av, l_bay, 0)
                    plot_beams.append([(x, y), (x + l_bay, y)])

        # --- SOLUCIÓN DE EIGENVALUES ---
        # Grados de libertad libres (excluyendo empotramientos en la base y rotaciones si se condensaran, aqui full)
        # Asumimos base empotrada: los nudos 1 a nx_nodes están fijos (GDL 0 a nx_nodes*3 - 1)
        fixed_dofs = np.arange(0, nx_nodes * 3)
        free_dofs = np.arange(nx_nodes * 3, total_dof)
        
        K_red = K_glob[np.ix_(free_dofs, free_dofs)]
        M_red = M_glob[np.ix_(free_dofs, free_dofs)]
        
        # Solver (eigh es para matrices simétricas, más rápido y estable)
        eigvals, eigvecs = np.linalg.eigh(K_red, M_red, b=None) # b=M_red si es generalized, pero aqui M es diagonal simple a veces
        # Usamos eig general para evitar problemas si M no es definida positiva perfecta numericamente
        vals_raw, _ = eig(K_red, M_red)
        vals_pos = np.sort(np.real(vals_raw))
        
        # Periodos
        w_sq = vals_pos[vals_pos > 1e-6] # Filtrar ruidos numéricos
        periods = 2 * np.pi / np.sqrt(w_sq)
        T1, T2 = periods[0], periods[1] # Fundamental y Segundo modo

        # --- CÁLCULO ESPECTRAL (ASCE 7-16) ---
        Fa, Fv = StructuralSolver.interpolar_sitio(ss_input, s1_input, site_class, 1.0, 1.0)
        SMS, SM1 = Fa * ss_input, Fv * s1_input
        SDS, SD1 = (2/3) * SMS, (2/3) * SM1
        Ts = SD1 / SDS
        T0 = 0.2 * Ts
        
        # Sa para T1
        if T1 < T0: Sa_design = SDS * (0.4 + 0.6 * T1 / T0)
        elif T1 < Ts: Sa_design = SDS
        elif T1 < tl_input: Sa_design = SD1 / T1
        else: Sa_design = (SD1 * tl_input) / (T1**2)

        total_weight_ton = (mass_nodes_count * mass_node) / 1000 * 9.81 # W = m*g (kN) -> Tonf (aprox /10)
        # Usaremos W en Toneladas fuerza directamente: Masa Total (kg) / 1000
        W_ton_mass = (mass_nodes_count * mass_node) / 1000
        Base_Shear = Sa_design * W_ton_mass # V = Sa * W (si Sa en g y W en ton)

        # ==========================================
        # 4. VISUALIZACIÓN DE RESULTADOS
        # ==========================================
        
        # Contenedores principales usando Tabs para limpieza
        tab1, tab2, tab3 = st.tabs([" Resumen Ejecutivo", " Modelo Gráfico", "Espectro Detallado"])

        with tab1:
            st.subheader("Resultados del Análisis Modal")
            
            # Fila de Métricas Principales con estilo destacado
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Periodo Fundamental ($T_1$)", f"{T1:.3f} s", delta="Modo 1")
            c2.metric("Periodo Secundario ($T_2$)", f"{T2:.3f} s", delta="Modo 2", delta_color="off")
            c3.metric("Cortante Basal ($V$)", f"{Base_Shear:.2f} Ton", help="Fuerza cortante total en la base")
            c4.metric("Coef. Sísmico ($C_s$)", f"{Sa_design:.3f} g", help="Aceleración de diseño")
            
            st.divider()
            
            # Sección de Rayleigh (Extraída del código original pero mejor presentada)
            w1, w2 = 2*np.pi/T1, 2*np.pi/T2
            rayleigh_coeffs = np.linalg.solve([[1/(2*w1), w1/2], [1/(2*w2), w2/2]], [zeta, zeta])
            
            col_info1, col_info2 = st.columns([2, 1])
            with col_info1:
                st.markdown("####  Parámetros de Diseño Calculados")
                st.dataframe({
                    "Parámetro": ["$F_a$", "$F_v$", "$S_{DS}$", "$S_{D1}$", "Peso Total"],
                    "Valor": [f"{Fa:.2f}", f"{Fv:.2f}", f"{SDS:.3f}", f"{SD1:.3f}", f"{W_ton_mass:.1f} Ton"],
                    "Unidad": ["-", "-", "g", "g", "Ton"]
                }, hide_index=True, use_container_width=True)
            
            with col_info2:
                st.info(f"""
                **Coeficientes de Amortiguamiento (Rayleigh):**
                
                $\\alpha = {rayleigh_coeffs[0]:.4f}$
                
                $\\beta = {rayleigh_coeffs[1]:.4f}$
                
                Estos valores son críticos para análisis tiempo-historia.
                """)

        with tab2:
            st.markdown("### Geometría y Discretización de Masas")
            # Gráfico del Pórtico Estilizado
            fig_frame, ax = plt.subplots(figsize=(10, 6))
            fig_frame.patch.set_facecolor(COLORS['background'])
            ax.set_facecolor(COLORS['background'])
            
            # Dibujar Vigas
            for line in plot_beams:
                ax.plot([line[0][0], line[1][0]], [line[0][1], line[1][1]], 
                        color=COLORS['accent'], linewidth=3, zorder=2)
            
            # Dibujar Columnas
            for line in plot_cols:
                ax.plot([line[0][0], line[1][0]], [line[0][1], line[1][1]], 
                        color=COLORS['secondary'], linewidth=4, zorder=1)
            
            # Dibujar Masas/Nudos
            x_vals = [n[0] for n in node_coords]
            y_vals = [n[1] for n in node_coords]
            # Filtramos nudos de la base para no ponerles circulos de masa grande
            x_mass = [x_vals[i] for i in range(len(y_vals)) if y_vals[i] > 0]
            y_mass = [y_vals[i] for i in range(len(y_vals)) if y_vals[i] > 0]
            
            ax.scatter(x_mass, y_mass, color=COLORS['highlight'], s=150, zorder=3, edgecolors='white', label='Masa Concentrada')
            ax.scatter(x_vals[:nx_nodes], y_vals[:nx_nodes], color='white', marker='s', s=100, zorder=3, label='Empotramiento')
            
            # Estética del gráfico
            ax.set_xlabel("Longitud (m)", color=COLORS['text'])
            ax.set_ylabel("Altura (m)", color=COLORS['text'])
            ax.tick_params(axis='x', colors=COLORS['text'])
            ax.tick_params(axis='y', colors=COLORS['text'])
            ax.spines['bottom'].set_color(COLORS['secondary'])
            ax.spines['left'].set_color(COLORS['secondary'])
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.grid(True, linestyle='--', color=COLORS['grid'], alpha=0.5)
            ax.legend(facecolor=COLORS['background'], edgecolor=COLORS['secondary'], labelcolor=COLORS['text'])
            ax.set_aspect('equal')
            
            st.pyplot(fig_frame)

        with tab3:
            # Gráfico del Espectro
            T_plot = np.linspace(0.0, 4.0, 300)
            Sa_plot = []
            for t in T_plot:
                if t < T0: val = SDS * (0.4 + 0.6 * t / T0)
                elif t < Ts: val = SDS
                elif t < tl_input: val = SD1 / t
                else: val = (SD1 * tl_input) / (t**2)
                Sa_plot.append(val)
            
            fig_spec, ax_s = plt.subplots(figsize=(10, 4))
            fig_spec.patch.set_facecolor(COLORS['background'])
            ax_s.set_facecolor(COLORS['background'])
            
            ax_s.plot(T_plot, Sa_plot, color=COLORS['text'], linewidth=2, label="Espectro ASCE 7-16")
            
            # Puntos T1 y T2
            ax_s.plot([T1, T1], [0, Sa_design], linestyle='--', color=COLORS['highlight'], alpha=0.7)
            ax_s.scatter(T1, Sa_design, color=COLORS['highlight'], s=100, zorder=5, label=f"T1 ({T1:.2f}s)")
            
            # Buscar Sa para T2 para graficarlo
            if T2 < T0: Sa_T2 = SDS * (0.4 + 0.6 * T2 / T0)
            elif T2 < Ts: Sa_T2 = SDS
            else: Sa_T2 = SD1 / T2 
            
            ax_s.scatter(T2, Sa_T2, color=COLORS['accent'], s=80, marker='^', zorder=5, label=f"T2 ({T2:.2f}s)")

            ax_s.set_title("Espectro de Diseño Elástico", color=COLORS['text'], fontweight='bold')
            ax_s.set_xlabel("Periodo T (s)", color=COLORS['text'])
            ax_s.set_ylabel("Aceleración Espectral Sa (g)", color=COLORS['text'])
            ax_s.tick_params(colors=COLORS['text'])
            ax_s.grid(True, color=COLORS['grid'], linestyle=':', alpha=0.6)
            ax_s.legend(facecolor=COLORS['background'], labelcolor=COLORS['text'])
            ax_s.spines['bottom'].set_color(COLORS['secondary'])
            ax_s.spines['left'].set_color(COLORS['secondary'])
            ax_s.spines['top'].set_visible(False)
            ax_s.spines['right'].set_visible(False)
            
            st.pyplot(fig_spec)
            
            st.markdown("""
            > **Nota Técnica:** El espectro se ha truncado a 4.0s para mejor visualización. 
            > El punto rojo indica el periodo fundamental usado para el cálculo del cortante basal.
            """)

else:
    # Estado inicial (Landing Page limpia)
    st.info(" Configura los parámetros en el panel lateral y presiona 'EJECUTAR ANÁLISIS' para comenzar.")
    
    # Un pequeño gráfico placeholder bonito
    x = np.linspace(0, 10, 100)
    y = np.sin(x) * np.exp(-0.1*x)
    fig_intro, ax_i = plt.subplots(figsize=(10, 2))
    fig_intro.patch.set_alpha(0.0)
    ax_i.axis('off')
    ax_i.plot(x, y, color=COLORS['secondary'], alpha=0.3)
    st.pyplot(fig_intro)