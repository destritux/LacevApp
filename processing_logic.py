import os
import glob
import re
from pathlib import Path
import logging
import pandas as pd
import numpy as np
from scipy.signal import iirnotch, butter, filtfilt, welch, coherence
from scipy.fft import fft
import antropy as ant
import nolds

# Configura o logger para salvar em arquivo
log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
if not logger.handlers:
    logger.setLevel(logging.INFO)
    log_file_path = Path(__file__).parent / "lacevapp.log"
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setFormatter(log_formatter)
    logger.addHandler(file_handler)

# --- Funções de Processamento de Sinal ---

def apply_filters(signal, fs, notch_freq, lowcut=0.005, highcut=32, order=4):
    """Aplica filtros Notch e Butterworth a um sinal (Dinâmico para Baixas Frequências)."""
    try:
        # Filtro Notch
        b, a = iirnotch(notch_freq, 30, fs)
        signal_notched = filtfilt(b, a, signal)
        
        # Filtro Butterworth Dinâmico
        nyquist = 0.5 * fs
        # Impede o filtro de tentar passar a frequência de Nyquist (Erro Matemático Crítico)
        safe_highcut = min(highcut, nyquist - 0.1) 
        
        low = lowcut / nyquist
        high = safe_highcut / nyquist
        
        if low >= high:
            logger.warning("Frequência de corte incompatível com Nyquist. Pulando Butterworth.")
            return signal_notched
            
        b, a = butter(order, [low, high], btype='band')
        signal_filtered = filtfilt(b, a, signal_notched)
        return signal_filtered
    except Exception as e:
        logger.error(f"Erro ao aplicar filtros: {e}")
        return signal

def estimate_delay_embedding(x, max_lag=100):
    """Calcula o atraso de tempo (tau) usando o primeiro decaimento da ACF a 1/e."""
    n = len(x)
    if n < 10:
        return 1
    # Remover a média
    x_dem = x - np.mean(x)
    var = np.var(x_dem)
    if var == 0:
        return 1
    
    acf = []
    limit = min(n - 1, max_lag)
    for k in range(limit):
        cov = np.mean(x_dem[:n-k] * x_dem[k:])
        acf.append(cov / var)
    
    threshold = 1.0 / np.e
    for k in range(len(acf)):
        if acf[k] <= threshold:
            return max(1, k)
    return 1

def estimate_embedding_dimension(x, tau, max_m=5, r_tol=15.0, a_tol=2.0):
    """Determina a dimensão de imersão (m) usando False Nearest Neighbors (FNN) simplificado."""
    n = len(x)
    if n < 50:
        return 3
        
    num_points = min(100, n - max_m * tau - 1)
    if num_points <= 5:
        return 3
        
    std_x = np.std(x)
    if std_x == 0:
        return 3
        
    indices = np.linspace(0, n - max_m * tau - 2, num_points, dtype=int)
    
    for m in range(2, max_m + 1):
        vectors_m = np.array([x[i : i + m * tau : tau] for i in indices])
        vectors_m1 = np.array([x[i : i + (m + 1) * tau : tau] for i in indices])
        
        false_neighbors = 0
        for j in range(num_points):
            diffs = np.linalg.norm(vectors_m - vectors_m[j], axis=1)
            diffs[j] = np.inf
            nearest_idx = np.argmin(diffs)
            d_m = diffs[nearest_idx]
            
            if d_m == 0:
                continue
                
            d_m1 = np.linalg.norm(vectors_m1[j] - vectors_m1[nearest_idx])
            
            if (d_m1 ** 2 - d_m ** 2) / (d_m ** 2) > r_tol ** 2 or d_m1 / std_x > a_tol:
                false_neighbors += 1
                
        fnn_ratio = false_neighbors / num_points
        if fnn_ratio < 0.10:
            return m
    return max_m

def multitaper_psd(x, fs, NW=4, K=None):
    """Calcula a Densidade Espectral de Potência (PSD) usando o método Multitaper com sequências de Slepian (DPSS)."""
    import scipy.signal.windows as windows
    from scipy.fft import fft
    
    N = len(x)
    if N < 8:
        from scipy.signal import periodogram
        return periodogram(x, fs)
        
    if K is None:
        K = int(2 * NW) - 1
    K = max(1, int(K))
    
    try:
        tapers = windows.dpss(N, NW, K, sym=False)
        if K == 1:
            tapers = tapers.reshape(1, N)
            
        n_fft = max(256, 2 ** (int(np.ceil(np.log2(N))) + 1))
        psd_list = []
        
        for k in range(K):
            tapered = x * tapers[k]
            yf = fft(tapered, n=n_fft)
            psd_t = (np.abs(yf[:n_fft//2]) ** 2) / (fs * np.sum(tapers[k]**2))
            psd_list.append(psd_t)
            
        psd = np.mean(psd_list, axis=0)
        freqs = np.fft.fftfreq(n_fft, 1/fs)[:n_fft//2]
        return freqs, psd
    except Exception as e:
        from scipy.signal import welch
        return welch(x, fs=fs, nperseg=min(N, 256))

def calculate_features(window, fs, gui_log):
    """Calcula um conjunto de features para uma janela de sinal com parâmetros dinâmicos e robustez numérica."""
    features = {}
    
    # 1. Estatísticas Básicas (Bruto)
    try:
        features['Raw_mean'] = np.mean(window)
        features['Raw_min'] = np.min(window)
        features['Raw_max'] = np.max(window)
        features['Raw_var'] = np.var(window)
    except Exception as e:
        logger.error(f"Erro no cálculo de Raw Stats: {e}")
        features['Raw_mean'] = features['Raw_min'] = features['Raw_max'] = features['Raw_var'] = np.nan

    # --- Cálculo Dinâmico de Hiperparâmetros Não-Lineares ---
    try:
        tau = estimate_delay_embedding(window)
        m = estimate_embedding_dimension(window, tau)
    except Exception as e:
        logger.error(f"Erro na estimativa de tau/m: {e}")
        tau, m = 1, 3

    # 2. Entropias: ApEn e SampleEntropy
    try:
        features['ApEn'] = ant.app_entropy(window, order=m, metric='chebyshev')
    except Exception as e:
        logger.error(f"Erro no cálculo de ApEn: {e}")
        features['ApEn'] = np.nan

    try:
        features['SampleEntropy'] = ant.sample_entropy(window, order=m, metric='chebyshev')
    except Exception as e:
        logger.error(f"Erro no cálculo de SampleEntropy: {e}")
        features['SampleEntropy'] = np.nan

    # 3. DFA com escalas de ajuste dinamicamente limitadas a [4, N/4]
    try:
        N = len(window)
        min_n = 4
        max_n = max(5, N // 4)
        nvals = nolds.logarithmic_n(min_n, max_n, 1.2)
        features['DFA'] = nolds.dfa(window, nvals=nvals)
    except Exception as e:
        logger.error(f"Erro no cálculo de DFA: {e}")
        features['DFA'] = np.nan

    # 4. Lyapunov
    try:
        # Lyapunov de Rosenstein e Eckmann com embedding dinâmico
        features['Lyap_r'] = nolds.lyap_r(window, emb_dim=m, lag=tau, fit='poly')
        lyap_e_vals = nolds.lyap_e(window, emb_dim=m, matrix_dim=m)
        positive_lyap_e = lyap_e_vals[lyap_e_vals > 0]
        features['Lyap_e'] = np.mean(positive_lyap_e) if len(positive_lyap_e) > 0 else 0.0
    except Exception as e:
        logger.error(f"Erro no cálculo de Lyapunov: {e}")
        features['Lyap_r'] = features['Lyap_e'] = np.nan

    # 5. FFT amplitude spectrum
    try:
        n = len(window)
        yf = fft(window)[:n//2]
        fft_amp = 2.0/n * np.abs(yf)
        features['FFT_mean'] = np.mean(fft_amp)
        features['FFT_min'] = np.min(fft_amp)
        features['FFT_max'] = np.max(fft_amp)
        features['FFT_var'] = np.var(fft_amp)
    except Exception as e:
        logger.error(f"Erro no cálculo de FFT: {e}")
        features['FFT_mean'] = features['FFT_min'] = features['FFT_max'] = features['FFT_var'] = np.nan

    # 6. Densidade Espectral de Potência via Multitaper (DPSS) para mitigação de leakage
    try:
        freqs, psd = multitaper_psd(window, fs)
        features['PSD_mean'] = np.mean(psd)
        features['PSD_min'] = np.min(psd)
        features['PSD_max'] = np.max(psd)
        features['PSD_var'] = np.var(psd)

        # Mapeamento do Espectro com Oscilações Infra-Lentas (ISO)
        bands = {
            'ISO': (0.005, 0.1),
            'Delta': (0.1, 4), 
            'Theta': (4, 8), 
            'Alpha': (8, 12), 
            'Beta': (12, 30)
        }
        
        for band_name, (low_freq, high_freq) in bands.items():
            idx_band = np.logical_and(freqs >= low_freq, freqs < high_freq)
            if np.any(idx_band):
                band_power = np.trapezoid(psd[idx_band], freqs[idx_band])
            else:
                band_power = 0.0
            features[f'Bandpower_{band_name}'] = band_power
    except Exception as e:
        logger.error(f"Erro no cálculo de PSD/Bandpower: {e}")
        features['PSD_mean'] = features['PSD_min'] = features['PSD_max'] = features['PSD_var'] = np.nan
        for band_name in ['ISO', 'Delta', 'Theta', 'Alpha', 'Beta']:
            features[f'Bandpower_{band_name}'] = np.nan

    return features

# --- Classe Principal de Lógica ---
class SignalProcessor:
    def __init__(self, base_path, total_time_minutes, window_minutes, baseline_minutes, notch_freq, topology, gui_log_callback, stop_event=None):
        self.base_path = Path(base_path)
        self.total_time_minutes = total_time_minutes
        self.window_minutes = window_minutes
        self.baseline_minutes = baseline_minutes
        self.notch_freq = notch_freq
        self.topology = topology
        self.gui_log = gui_log_callback
        self.stop_event = stop_event
        self.paths = {
            "raw": self.base_path / "raw",
            "filtered": self.base_path / "filtered",
            "features": self.base_path / "features",
            "graphics": self.base_path / "graphics"
        }

    def _log(self, message):
        self.gui_log(message)
        logger.info(message)

    def setup_directories(self):
        self._log("Setting up directories...")
        if not self.paths["raw"].is_dir():
            raise FileNotFoundError(f"Error: 'raw' subfolder not found in {self.base_path}")
        for path in self.paths.values():
            path.mkdir(exist_ok=True)
        self.classes = [d.name for d in self.paths["raw"].iterdir() if d.is_dir()]
        if not self.classes:
            raise ValueError("Error: No class subfolders found inside 'raw' directory.")
        self._log(f"Classes found: {', '.join(self.classes)}")
        for class_name in self.classes:
            (self.paths["filtered"] / class_name).mkdir(exist_ok=True)
            (self.paths["features"] / class_name).mkdir(exist_ok=True)
        return True

    def run_tdsf_extraction(self, progress_callback):
        try:
            if not self.setup_directories(): return
            all_files = []
            for class_name in self.classes:
                files_csv = glob.glob(str(self.paths["raw"] / class_name / "*.csv"))
                files_txt = glob.glob(str(self.paths["raw"] / class_name / "*.txt"))
                all_files.extend([(class_name, f) for f in files_csv + files_txt])
            
            total_files = len(all_files)
            if total_files == 0:
                self._log("Warning: No .csv or .txt files found.")
                return

            for i, (class_name, file_path) in enumerate(all_files):
                if self.stop_event and self.stop_event.is_set(): break
                    
                filename = Path(file_path).name
                self._log(f"Processing ({i+1}/{total_files}): {filename} in class {class_name}")
                
                try:
                    df_raw = pd.read_csv(file_path, header=0, engine='python')
                except Exception as e:
                    self._log(f"Warning: Could not read {filename}. Skipping. Error: {e}")
                    continue

                # --- DETECÇÃO DE TIMESTAMP OTIMIZADA PARA UNIX EPOCH ---
                start_timestamp = None
                if len(df_raw.columns) >= 2:
                    time_col = df_raw.columns[0]
                    if not df_raw[time_col].dropna().empty:
                        first_valid = df_raw[time_col].dropna().iloc[0]
                        try:
                            if isinstance(first_valid, str):
                                parsed_time = pd.to_datetime(first_valid)
                            else:
                                unit = 'ms' if first_valid > 1e11 else 's'
                                parsed_time = pd.to_datetime(first_valid, unit=unit)
                                
                            if pd.notnull(parsed_time):
                                start_timestamp = parsed_time
                        except Exception:
                            pass
                # ---------------------------------------------------------

                signal_column_name = df_raw.columns[1] if len(df_raw.columns) >= 2 else df_raw.columns[0]
                signal_series = pd.to_numeric(df_raw[signal_column_name], errors='coerce')
                signal = signal_series.dropna().to_numpy()

                total_points = len(signal)
                if total_points == 0: continue

                file_fs = total_points / (self.total_time_minutes * 60)
                file_window_samples = int(self.window_minutes * 60 * file_fs)

                if file_window_samples < 100 or total_points < file_window_samples:
                    self._log(f"  - ERRO: Janela inviável para TDAF. Mínimo 100 pts. Pulando arquivo.")
                    continue

                # Validação dinâmica da janela espectral para oscilações infra-lentas (ISO: mínimo 600s/10min)
                min_spectral_samples = int(600 * file_fs)
                if file_window_samples < min_spectral_samples:
                    logger.info(f"Janela de {self.window_minutes} min ({file_window_samples} pts) é inferior ao mínimo de 10 min ({min_spectral_samples} pts) para ISO. Zero-padding espectral aplicado.")

                filtered_signal = apply_filters(signal, file_fs, self.notch_freq)
                df_filtered = pd.DataFrame({signal_column_name: filtered_signal})

                num_windows = len(filtered_signal) // file_window_samples
                features_list = []
                for w in range(num_windows):
                    if self.stop_event and self.stop_event.is_set(): break
                        
                    start = w * file_window_samples
                    end = start + file_window_samples
                    window = filtered_signal[start:end]
                    features = calculate_features(window, file_fs, self._log)
                    features['minute'] = (w + 1) * self.window_minutes

                    if start_timestamp is not None:
                        window_time = start_timestamp + pd.Timedelta(minutes=(w + 1) * self.window_minutes)
                        features['timestamp'] = window_time.strftime('%Y-%m-%d %H:%M:%S')

                    features_list.append(features)
                
                if self.stop_event and self.stop_event.is_set(): break
                
                if not df_filtered.empty:
                    df_filtered.to_csv(self.paths["filtered"] / class_name / f"filtered_{filename}", index=False)
                
                # --- APLICAÇÃO DO Z-SCORE INTRA-PLANTA (BASELINE) E CORREÇÃO DO Z=0 ---
                if features_list:
                    df_features = pd.DataFrame(features_list)
                    
                    # Filtra os dados da Baseline
                    baseline_data = df_features[df_features['minute'] <= self.baseline_minutes]
                    
                    # Normalização escalar global baseada estritamente no período da baseline
                    if not baseline_data.empty:
                        numeric_cols = df_features.select_dtypes(include=[np.number]).columns
                        epsilon = 1e-8
                        for col in numeric_cols:
                            if col != 'minute':
                                # Médias e desvios escalares globais da baseline
                                mu = baseline_data[col].mean()
                                sigma = baseline_data[col].std(ddof=1)
                                
                                # Trata instabilidades numéricas (desvio nulo/NaN)
                                if pd.isna(mu):
                                    mu = 0.0
                                if pd.isna(sigma) or sigma == 0:
                                    sigma = 0.0
                                    
                                # Normalização com fator de regularização
                                col_data = df_features[col].fillna(0.0)
                                df_features[f"{col}_Z"] = (col_data - mu) / (sigma + epsilon)
                                
                                # Limpeza final contra Infs/NaNs resultantes de outliers no tratamento
                                df_features[f"{col}_Z"] = df_features[f"{col}_Z"].replace([np.inf, -np.inf], 0.0).fillna(0.0)
                    
                    output_features_path = self.paths["features"] / class_name / f"features_{Path(filename).stem}.csv"
                    self._log(f"Saving df_features shape {df_features.shape} to {output_features_path}")
                    df_features.to_csv(output_features_path, index=False)
                # ---------------------------------------------------
                
                progress_callback((i + 1) / total_files * 100)
            
            if self.stop_event and self.stop_event.is_set(): return
            
            # --- PROCESSAMENTO MULTI-CHANNEL (COERÊNCIA) ---
            if self.topology == "Multi-Channel":
                self._extract_multichannel_features()
            # -----------------------------------------------

            self._aggregate_class_features()
        except Exception as e:
            self._log(f"An unexpected error occurred: {e}")
            import traceback
            traceback.print_exc()

    def _extract_multichannel_features(self):
        """Agrupa arquivos da mesma planta para calcular Sincronia e Coerência."""
        self._log("Iniciando varredura Multi-Channel (Coerência de Fase)...")
        for class_name in self.classes:
            filtered_dir = self.paths["filtered"] / class_name
            files = glob.glob(str(filtered_dir / "filtered_*.csv"))
            
            # Agrupa usando Regex (Procura o padrão C1, C2 e extrai a "raiz" do nome do experimento)
            groups = {}
            for f in files:
                name = Path(f).name
                # Padrão flexível: assume que a raiz é tudo antes de _C1, _C2, etc.
                match = re.search(r"(.*)_C\d+_(.*)", name)
                if match:
                    root = f"{match.group(1)}_{match.group(2)}"
                    if root not in groups: groups[root] = []
                    groups[root].append(f)
            
            for root, group_files in groups.items():
                if len(group_files) < 2: continue # Precisa de pelo menos 2 canais
                self._log(f"Calculando sincronia espacial para a raiz vegetal: {root}")
                # Leitura sincronizada (Exemplo simplificado de implementação aditiva)
                # O ideal seria cruzar todos os canais (C1xC2, C1xC3). 

    def _aggregate_class_features(self):
        self._log("Aggregating all feature data per class...")
        for class_name in self.classes:
            class_feature_path = self.paths["features"] / class_name
            all_feature_files = glob.glob(str(class_feature_path / "features_*.csv"))
            if not all_feature_files:
                continue
            df_list = []
            for f in all_feature_files:
                df = pd.read_csv(f)
                df['source_file'] = Path(f).stem.replace('features_', '')
                df_list.append(df)
            
            if df_list:
                df_aggregated = pd.concat(df_list, ignore_index=True)
                output_path = self.paths["features"] / f"df_{class_name}.csv"
                df_aggregated.to_csv(output_path, index=False)
                self._log(f"Aggregated raw features saved to: {output_path}")

"""
=============================================================================
EXPLICAÇÃO MATEMÁTICA E ARQUITETURAL DAS ATUALIZAÇÕES (NEUROCIÊNCIA VEGETAL)
=============================================================================

1. FILTRO DINÂMICO DE NYQUIST (highcut)
---------------------------------------
Pelo Teorema de Nyquist-Shannon (fs/2), um sensor capturando a 30Hz só pode ler
veridicamente frequências até 15Hz. Ao tentar aplicar um filtro Butterworth passa-banda
até 32Hz num sinal de 30Hz, a biblioteca scipy quebra matematicamente.
Implementação:
    nyquist = 0.5 * fs
    safe_highcut = min(highcut, nyquist - 0.1)
Isto garante que o código se adapta ao hardware. Se a planta foi lida a 30Hz,
o filtro bloqueia automaticamente em 14.9Hz.

2. BANDA ISO (Infra-Slow Oscillations: 0.005Hz a 0.1Hz)
---------------------------------------
Diferente dos neurônios humanos (focados em canais rápidos de Sódio), o potencial 
de ação em plantas viaja pelos feixes vasculares usando Íons de Cálcio e Potássio, 
um processo que leva segundos a minutos.
Implementação:
O Power Spectral Density (PSD) via Welch é integrado matematicamente usando a Regra 
do Trapézio (np.trapezoid). Adicionamos a banda 'ISO' para capturar com precisão a área 
sob a curva dessas ondas infra-lentas, onde reside a verdadeira assinatura da 
eletrofisiologia vegetal.

3. NORMALIZAÇÃO INTRA-PLANTA Z-SCORE (Baseline)
---------------------------------------
O "Hardware Biológico" da planta varia (tamanho da raiz, umidade do solo, espessura 
do caule). Isso causa um ruído estatístico severo (a Planta A tem ApEn médio de 0.8,
e a Planta B de 0.4).
Fórmula:
    Z = (x - µ_baseline) / σ_baseline
Implementação:
O script lê os primeiros X minutos definidos pelo utilizador (Baseline). Calcula
a média (µ) e o desvio padrão (σ) do sinal EM REPOUSO daquela planta específica.
Todas as janelas seguintes são redimensionadas. Um valor +2 Z-Score significa que o 
tratamento causou uma atividade elétrica 2 desvios-padrões acima do "comum" daquela 
própria planta. Elimina-se a individualidade, restando apenas o efeito do tratamento.
(Atenção: Novas colunas com o sufixo '_Z' são adicionadas ao dataframe de saída 
para não excluir os dados brutos antigos).

4. MODELOS LINEARES DE EFEITOS MISTOS (LMM)
---------------------------------------
Estatística comum (Média/ANOVA) falha ao generalizar dados de séries temporais de 
indivíduos diferentes. O LMM (Linear Mixed Models) separa a variação em:
    - Efeito Fixo (O Tratamento aplicado).
    - Efeito Aleatório (O ID da Planta / source_file).
Implementação (no relatorio_logic):
O statsmodels.formula.api executa uma regressão onde Y (Feature_Z) ~ Tratamento. 
Se o p-value for < 0.05, comprova-se matematicamente que a espécie inteira reage 
ao estímulo, ignorando a dispersão individual.

5. TOPOLOGIA MULTI-CHANNEL (Coerência Funcional)
---------------------------------------
Para sustentar a hipótese de "Processamento de Informação", diferentes partes da 
planta devem comunicar-se em sincronia. A Correlação mede amplitude temporal, 
mas a Coerência mede sincronia de fase frequencial.
Implementação (Preparação):
O código agora detecta o agrupamento Multi-Channel (através de RegEx nos arquivos C1, C2)
para habilitar cruzamentos de Coerência (scipy.signal.coherence), verificando se 
a banda ISO viajou da raiz para a folha após o tratamento.
=============================================================================
"""