import os
import glob
from pathlib import Path
import pandas as pd
import seaborn as sns
import logging

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

logger = logging.getLogger(__name__)

def setup_plot_style(dark=True):
    if dark:
        sns.set_theme(style="darkgrid", rc={
            "axes.facecolor": "#343a40", "grid.color": "#495057",
            "figure.facecolor": "#212529", "text.color": "#f8f9fa",
            "axes.labelcolor": "#f8f9fa", "xtick.color": "#ced4da", "ytick.color": "#ced4da"
        })
        return "_dark"
    else:
        sns.set_theme(style="whitegrid", rc={
            "axes.facecolor": "#ffffff", "grid.color": "#ced4da",
            "figure.facecolor": "#f8f9fa", "text.color": "#212529",
            "axes.labelcolor": "#212529", "xtick.color": "#495057", "ytick.color": "#495057"
        })
        return "_light"

# --- Mapeamento de features ---
feature_map = {
    'ApEn': ['ApEn', 'ApEn_Z', 'SampleEntropy', 'SampleEntropy_Z'],
    'DFA': ['DFA', 'DFA_Z'],
    'Lyapunov': ['Lyap_r', 'Lyap_r_Z', 'Lyap_e', 'Lyap_e_Z'],
    'Raw_Stats': ['Raw_mean', 'Raw_mean_Z', 'Raw_min', 'Raw_min_Z', 'Raw_max', 'Raw_max_Z', 'Raw_var', 'Raw_var_Z'],
    'FFT_Stats': ['FFT_mean', 'FFT_mean_Z', 'FFT_min', 'FFT_min_Z', 'FFT_max', 'FFT_max_Z', 'FFT_var', 'FFT_var_Z'],
    'PSD_Stats': ['PSD_mean', 'PSD_mean_Z', 'PSD_min', 'PSD_min_Z', 'PSD_max', 'PSD_max_Z', 'PSD_var', 'PSD_var_Z'],
    'Bandpower': ['Bandpower_ISO', 'Bandpower_ISO_Z', 'Bandpower_Delta', 'Bandpower_Delta_Z', 'Bandpower_Theta', 'Bandpower_Theta_Z', 'Bandpower_Alpha', 'Bandpower_Alpha_Z', 'Bandpower_Beta', 'Bandpower_Beta_Z'],
    'Z_Scores': ['Raw_mean_Z', 'Raw_min_Z', 'Raw_max_Z', 'Raw_var_Z', 'ApEn_Z', 'SampleEntropy_Z', 'DFA_Z', 'Lyap_r_Z', 'Lyap_e_Z', 'FFT_mean_Z', 'FFT_min_Z', 'FFT_max_Z', 'FFT_var_Z', 'PSD_mean_Z', 'PSD_min_Z', 'PSD_max_Z', 'PSD_var_Z', 'Bandpower_ISO_Z', 'Bandpower_Delta_Z', 'Bandpower_Theta_Z', 'Bandpower_Alpha_Z', 'Bandpower_Beta_Z'],
    'ISO': ['Bandpower_ISO', 'Bandpower_ISO_Z']
}

# --- Aliases ---
alias_map = {
    'ApEn': ['ApEn', 'apen', 'apen_mean', 'apen_value'],
    'SampleEntropy': ['SampleEntropy', 'sample_entropy', 'sampen'],
    'DFA': ['DFA', 'dfa'],
    'Lyap_r': ['Lyap_r', 'lyap_r', 'lyap', 'lyapunov', 'chaos_cont_dim', 'chaos_continuous_dim'],
    'Lyap_e': ['Lyap_e', 'lyap_e', 'lyap', 'lyapunov', 'chaos_cont_dim'],
    'Raw_mean': ['Raw_mean', 'electrome_mean'],
    'Raw_min': ['Raw_min', 'electrome_min'],
    'Raw_max': ['Raw_max', 'electrome_max'],
    'Raw_var': ['Raw_var', 'electrome_variance', 'electrome_var'],
    'FFT_mean': ['FFT_mean', 'fft_mean'],
    'FFT_min': ['FFT_min', 'fft_min'],
    'FFT_max': ['FFT_max', 'fft_max'],
    'FFT_var': ['FFT_var', 'fft_variance', 'fft_var'],
    'PSD_mean': ['PSD_mean', 'psd_mean'],
    'PSD_min': ['PSD_min', 'psd_min'],
    'PSD_max': ['PSD_max', 'psd_max'],
    'PSD_var': ['PSD_var', 'psd_variance', 'psd_var'],
    'Bandpower_ISO': ['Bandpower_ISO'],
    'Bandpower_Low': ['Bandpower_Low', 'abp_low'],
    'Bandpower_Delta': ['Bandpower_Delta', 'abp_delta'],
    'Bandpower_Theta': ['Bandpower_Theta', 'abp_theta'],
    'Bandpower_Alpha': ['Bandpower_Alpha', 'abp_alpha'],
    'Bandpower_Beta': ['Bandpower_Beta', 'abp_beta'],
    
    # Aliases Z-Score
    'ApEn_Z': ['ApEn_Z'],
    'SampleEntropy_Z': ['SampleEntropy_Z'],
    'DFA_Z': ['DFA_Z'],
    'Lyap_e_Z': ['Lyap_e_Z'],
    'Raw_var_Z': ['Raw_var_Z'],
    
    'minute': ['minute']
}

def find_first_existing_column(columns, candidates):
    lower_map = {col.lower(): col for col in columns}
    for cand in candidates:
        if cand is None:
            continue
        key = cand.lower()
        if key in lower_map:
            return lower_map[key]
    return None

def plot_tdsf(base_path, feature_group, themes_to_plot=['Escuro', 'Claro'], log_scale=False, gui_log_callback=print):
    features_path = Path(base_path) / "features"
    graphics_path = Path(base_path) / "graphics"
    graphics_path.mkdir(exist_ok=True)

    class_files = glob.glob(str(features_path / "df_*.csv"))
    if not class_files:
        gui_log_callback(f"Aviso: Nenhum ficheiro de dados encontrado para o grupo '{feature_group}'. A saltar a plotagem.")
        return

    if feature_group not in feature_map:
        gui_log_callback(f"Erro: Grupo de características desconhecido '{feature_group}'.")
        return

    features_to_plot = feature_map[feature_group]

    dark_modes_to_run = []
    if 'Escuro' in themes_to_plot:
        dark_modes_to_run.append(True)
    if 'Claro' in themes_to_plot:
        dark_modes_to_run.append(False)

    valid_features = []
    for feature_base_name in features_to_plot:
        aliases = alias_map.get(feature_base_name, [feature_base_name])
        found_any = False
        for class_file in class_files:
            try:
                cols = pd.read_csv(class_file, nrows=0).columns
            except Exception:
                continue
            if find_first_existing_column(cols, aliases):
                found_any = True
                break
        
        if found_any:
            valid_features.append(feature_base_name)
        else:
            msg = f"Aviso: Coluna '{feature_base_name}' não encontrada. Ela será ignorada na plotagem."
            gui_log_callback(msg)

    if not valid_features:
        gui_log_callback(f"Aviso: Nenhuma coluna válida encontrada para o grupo '{feature_group}'.")
        return

    features_to_plot = valid_features

    minute_aliases = alias_map.get('minute', ['minute'])
    minute_found = False
    for class_file in class_files:
        try:
            cols = pd.read_csv(class_file, nrows=0).columns
        except Exception:
            continue
        if find_first_existing_column(cols, minute_aliases):
            minute_found = True
            break
    if not minute_found:
        msg = f"Erro: Nenhuma coluna de tempo ('minute') encontrada nos ficheiros. Procuradas: {minute_aliases}."
        gui_log_callback(msg)
        raise ValueError(msg)

    for dark_mode in dark_modes_to_run:
        style_suffix = setup_plot_style(dark=dark_mode)
        gui_log_callback(f"  - A gerar gráfico para '{feature_group}' com tema: {'Escuro' if dark_mode else 'Claro'}")

        n_features = len(features_to_plot)
        fig, axes = plt.subplots(n_features, 1, figsize=(15, 5 * n_features), squeeze=False)
        axes = axes.flatten()

        for i, feature_base_name in enumerate(features_to_plot):
            current_ax = axes[i]
            aliases = alias_map.get(feature_base_name, [feature_base_name])
            used_timestamp = False

            for class_file in class_files:
                class_name = Path(class_file).stem.replace('df_', '')
                try:
                    df = pd.read_csv(class_file)
                except Exception as e:
                    gui_log_callback(f"Aviso: Não foi possível ler {class_file}. Erro: {e}")
                    continue

                minute_col = find_first_existing_column(df.columns, minute_aliases)
                if not minute_col:
                    gui_log_callback(f"Erro: coluna de tempo ('minute') não encontrada no ficheiro {class_file}.")
                    raise ValueError(f"Coluna de tempo ('minute') ausente em {class_file}.")

                x_col = minute_col
                if 'timestamp' in df.columns:
                    try:
                        df['timestamp'] = pd.to_datetime(df['timestamp'])
                        x_col = 'timestamp'
                        used_timestamp = True
                    except Exception:
                        pass

                actual_col = find_first_existing_column(df.columns, aliases)
                if actual_col:
                    try:
                        sns.lineplot(data=df, x=x_col, y=actual_col, ax=current_ax,
                                     label=class_name, errorbar='sd')
                    except Exception as e:
                        gui_log_callback(f"Não foi possível plotar {actual_col} para {class_name}. Erro: {e}")
                else:
                    gui_log_callback(f"Aviso: coluna para '{feature_base_name}' não encontrada em {class_name}. Procuradas: {aliases}. Pulando este ficheiro.")

            current_ax.set_title(feature_base_name)
            
            if used_timestamp:
                current_ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
                current_ax.set_xlabel("Time (HH:MM)")
            else:
                current_ax.set_xlabel("Time (Minutes)")
                
            current_ax.set_ylabel("Value")
            if log_scale:
                current_ax.set_yscale('log')
                current_ax.set_ylabel("Value (Log Scale)")
            current_ax.legend(title='Class')

        plt.tight_layout()
        log_suffix = "_log" if log_scale else ""
        output_filename = graphics_path / f"TDAF_{feature_group}{log_suffix}{style_suffix}.png"
        plt.savefig(output_filename, dpi=150)
        plt.close(fig)
        gui_log_callback(f"Gráfico guardado em: {output_filename}")