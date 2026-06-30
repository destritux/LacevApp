import pandas as pd
import numpy as np
from pathlib import Path
import glob
from PIL import Image
import scipy.stats
import scipy.interpolate

class TimeSeriesClusterTest:
    def __init__(self, n_permutations=1000, alpha_local=0.05, min_replicates=3):
        self.n_permutations = n_permutations
        self.alpha_local = alpha_local
        self.min_replicates = min_replicates

    def _fast_ttest_2groups(self, g1, g2):
        """Welch's independent t-test computed column-wise (vectorized)"""
        n1 = np.sum(~np.isnan(g1), axis=0)
        n2 = np.sum(~np.isnan(g2), axis=0)
        
        m1 = np.nanmean(g1, axis=0)
        m2 = np.nanmean(g2, axis=0)
        v1 = np.nanvar(g1, axis=0, ddof=1)
        v2 = np.nanvar(g2, axis=0, ddof=1)
        
        pooled_se = np.sqrt(v1 / n1 + v2 / n2)
        pooled_se = np.where(pooled_se == 0, 1e-10, pooled_se)
        
        t = (m1 - m2) / pooled_se
        
        # Welch-Satterthwaite degrees of freedom
        num = (v1/n1 + v2/n2)**2
        den = (v1/n1)**2 / (n1 - 1) + (v2/n2)**2 / (n2 - 1)
        den = np.where(den == 0, 1e-10, den)
        df = num / den
        df = np.maximum(df, 1)
        
        p = 2 * (1 - scipy.stats.t.cdf(np.abs(t), df))
        
        # Mask out columns with insufficient data
        mask = (n1 < self.min_replicates) | (n2 < self.min_replicates)
        t[mask] = np.nan
        p[mask] = np.nan
        return t, p

    def _fast_anova_multigroups(self, groups_list):
        """One-Way ANOVA computed column-wise across multiple group matrices (vectorized)"""
        k = len(groups_list)
        n_points = groups_list[0].shape[1]
        
        n_g = np.array([np.sum(~np.isnan(g), axis=0) for g in groups_list]) # shape (k, n_points)
        m_g = np.array([np.nanmean(g, axis=0) for g in groups_list]) # shape (k, n_points)
        v_g = np.array([np.nanvar(g, axis=0, ddof=1) for g in groups_list]) # shape (k, n_points)
        
        total_n = np.sum(n_g, axis=0) # shape (n_points)
        # Avoid division by zero in grand mean
        total_n_safe = np.where(total_n == 0, 1, total_n)
        grand_mean = np.sum(n_g * m_g, axis=0) / total_n_safe # shape (n_points)
        
        ss_between = np.sum(n_g * (m_g - grand_mean)**2, axis=0)
        df_between = k - 1
        ms_between = ss_between / df_between
        
        ss_within = np.sum((n_g - 1) * v_g, axis=0)
        df_within = np.sum(n_g - 1, axis=0)
        df_within_safe = np.maximum(df_within, 1)
        ms_within = ss_within / df_within_safe
        
        # Avoid division by zero
        ms_within = np.where(ms_within == 0, 1e-10, ms_within)
        f_vals = ms_between / ms_within
        p_vals = scipy.stats.f.sf(f_vals, df_between, df_within_safe)
        
        # Mask out columns with insufficient data
        insufficient = np.any(n_g < self.min_replicates, axis=0)
        f_vals[insufficient] = np.nan
        p_vals[insufficient] = np.nan
        
        return f_vals, p_vals

    def find_clusters(self, stats, p_values):
        """Identifies contiguous significant clusters and computes their mass"""
        sig = (p_values < self.alpha_local) & (~np.isnan(p_values))
        clusters = []
        current_cluster = []
        
        for idx, is_sig in enumerate(sig):
            if is_sig:
                current_cluster.append(idx)
            else:
                if current_cluster:
                    clusters.append(current_cluster)
                    current_cluster = []
        if current_cluster:
            clusters.append(current_cluster)
            
        cluster_details = []
        for c in clusters:
            mass = np.nansum(np.abs(stats[c]))
            cluster_details.append({
                'indices': c,
                'start_idx': c[0],
                'end_idx': c[-1],
                'mass': mass
            })
        return cluster_details

    def run_permutation_test(self, groups_list, target_grid):
        """Runs the cluster permutation test for 2 or more groups"""
        # 1. Compute Observed Stats
        if len(groups_list) == 2:
            stat_obs, p_obs = self._fast_ttest_2groups(groups_list[0], groups_list[1])
        else:
            stat_obs, p_obs = self._fast_anova_multigroups(groups_list)
            
        obs_clusters = self.find_clusters(stat_obs, p_obs)
        if not obs_clusters:
            return [] # No clusters observed
            
        # 2. Build Null Distribution via Permutations
        combined = np.concatenate(groups_list, axis=0)
        group_sizes = [len(g) for g in groups_list]
        null_masses = []
        
        np.random.seed(42)
        for _ in range(self.n_permutations):
            # Shuffle labels by permuting the combined data matrix along axis 0
            shuffled = np.random.permutation(combined)
            
            # Split back into groups
            shuff_groups = []
            start = 0
            for size in group_sizes:
                shuff_groups.append(shuffled[start:start+size])
                start += size
                
            # Compute stats on permuted groups
            if len(groups_list) == 2:
                stat_shuff, p_shuff = self._fast_ttest_2groups(shuff_groups[0], shuff_groups[1])
            else:
                stat_shuff, p_shuff = self._fast_anova_multigroups(shuff_groups)
                
            shuff_clusters = self.find_clusters(stat_shuff, p_shuff)
            if shuff_clusters:
                null_masses.append(max(c['mass'] for c in shuff_clusters))
            else:
                null_masses.append(0.0)
                
        null_masses = np.array(null_masses)
        
        # 3. Correct Observed Cluster p-values
        results = []
        for idx, c in enumerate(obs_clusters):
            p_corrected = np.sum(null_masses >= c['mass']) / self.n_permutations
            results.append({
                'Cluster_ID': idx + 1,
                'Start_Index': int(c['start_idx']),
                'End_Index': int(c['end_idx']),
                'Start_Minute': float(target_grid[c['start_idx']]),
                'End_Minute': float(target_grid[c['end_idx']]),
                'Observed_Mass': float(c['mass']),
                'p_corrected': float(p_corrected)
            })
            
        return results


def generate_html_report(base_path, experiment_description, gui_log_callback):
    """Gera relatório HTML completo com estatísticas clássicas OLS, resumos CSV na pasta 'results' e gráficos."""
    base_path = Path(base_path)
    features_path = base_path / "features"
    graphics_path = base_path / "graphics"
    results_path = base_path / "results"
    
    # Cria pasta de resultados se não existir
    results_path.mkdir(exist_ok=True)

    gui_log_callback("Carregando dados processados...")
    class_files = glob.glob(str(features_path / "df_*.csv"))
    if not class_files:
        gui_log_callback("Erro: Nenhum ficheiro df_*.csv encontrado.")
        return

    all_data = {}
    master_df_list = []
    
    for f in class_files:
        class_name = Path(f).stem.replace('df_', '')
        df = pd.read_csv(f)

        drop_cols = [c for c in df.columns if c.lower().startswith("unnamed")
                     or c.lower() in ["classe", "nameclass", "date"]]
        df = df.drop(columns=drop_cols, errors="ignore")

        all_data[class_name] = df
        
        # Preparação do Master DF para regressão
        df_reg = df.copy()
        df_reg['TreatmentClass'] = class_name
        master_df_list.append(df_reg)

    gui_log_callback("Calculando estatísticas de resumo...")
    numeric_cols = all_data[list(all_data.keys())[0]].select_dtypes(include='number').columns.drop('minute', errors='ignore')
    summary_stats = {'Classe': []}
    for col in numeric_cols:
        summary_stats[f'{col}'] = []

    for class_name, df in all_data.items():
        summary_stats['Classe'].append(class_name)
        for col in numeric_cols:
            summary_stats[f'{col}'].append(df[col].mean())

    summary_df = pd.DataFrame(summary_stats).set_index('Classe')
    
    # Geração de Diferença Percentual para todos os pares classe a classe
    import itertools
    classes = sorted(all_data.keys())
    pairs = list(itertools.combinations(classes, 2))
    
    comparison_rows = []
    for classA, classB in pairs:
        row_data = {'Par': f"{classB} vs {classA}"}
        for col in numeric_cols:
            valA = summary_df.loc[classA, col]
            valB = summary_df.loc[classB, col]
            if valA != 0 and not pd.isna(valA) and valA is not None:
                row_data[col] = ((valB - valA) / valA) * 100
            else:
                row_data[col] = np.nan
        comparison_rows.append(row_data)
        
    comparison_df = pd.DataFrame(comparison_rows).set_index('Par')
    summary_df_transposed = summary_df.T
    comparison_df_transposed = comparison_df.T

    # Salva resumos estatísticos em formato CSV na pasta 'results'
    summary_df_transposed.to_csv(results_path / "summary_stats.csv")
    comparison_df_transposed.to_csv(results_path / "comparison_stats.csv")

    # --- MODELO LINEAR (OLS) PAREADO COM STATSMODELS ---
    ols_html = ""
    highlight_html = ""
    json_data_str = "{}"
    cluster_results_data = {}
    
    try:
        gui_log_callback("Tentando processar estatísticas de Regressão Linear (OLS) pareadas...")
        import statsmodels.formula.api as smf
        import json
        
        master_df = pd.concat(master_df_list, ignore_index=True)
        z_cols = [c for c in master_df.columns if c.endswith('_Z')]
        classes = sorted(all_data.keys())
        
        # Serializar dados vazios (os dados agora vêm do features_data.json)
        json_data_str = "{}"
        
        def _get_metric_url(metric_name):
            base_name = metric_name.replace('_Z', '')
            urls = {
                'Raw_mean': 'https://en.wikipedia.org/wiki/Mean',
                'Raw_min': 'https://en.wikipedia.org/wiki/Sample_maximum_and_minimum',
                'Raw_max': 'https://en.wikipedia.org/wiki/Sample_maximum_and_minimum',
                'Raw_var': 'https://en.wikipedia.org/wiki/Variance',
                'ApEn': 'https://en.wikipedia.org/wiki/Approximate_entropy',
                'SampleEntropy': 'https://en.wikipedia.org/wiki/Sample_entropy',
                'DFA': 'https://en.wikipedia.org/wiki/Detrended_fluctuation_analysis',
                'Lyap_r': 'https://en.wikipedia.org/wiki/Lyapunov_exponent',
                'Lyap_e': 'https://en.wikipedia.org/wiki/Lyapunov_exponent',
                'FFT_mean': 'https://en.wikipedia.org/wiki/Fast_Fourier_transform',
                'FFT_min': 'https://en.wikipedia.org/wiki/Fast_Fourier_transform',
                'FFT_max': 'https://en.wikipedia.org/wiki/Fast_Fourier_transform',
                'FFT_var': 'https://en.wikipedia.org/wiki/Fast_Fourier_transform',
                'PSD_mean': 'https://en.wikipedia.org/wiki/Spectral_density',
                'PSD_min': 'https://en.wikipedia.org/wiki/Spectral_density',
                'PSD_max': 'https://en.wikipedia.org/wiki/Spectral_density',
                'PSD_var': 'https://en.wikipedia.org/wiki/Spectral_density',
                'Bandpower_ISO': 'https://pubmed.ncbi.nlm.nih.gov/?term=infra-slow+oscillations+plants',
                'Bandpower_Delta': 'https://en.wikipedia.org/wiki/Delta_wave',
                'Bandpower_Theta': 'https://en.wikipedia.org/wiki/Theta_wave',
                'Bandpower_Alpha': 'https://en.wikipedia.org/wiki/Alpha_wave',
                'Bandpower_Beta': 'https://en.wikipedia.org/wiki/Beta_wave'
            }
            return urls.get(base_name, 'https://en.wikipedia.org/wiki/Spectral_analysis')

        if len(classes) >= 2 and z_cols:
            pairs = list(itertools.combinations(classes, 2))
            pairwise_results = []
            pair_summaries = []
            
            for classA, classB in pairs:
                # Filtrar o master_df para conter apenas observações desses dois grupos
                pair_df = master_df[master_df['TreatmentClass'].isin([classA, classB])].copy()
                pair_df['TreatmentClass'] = pd.Categorical(pair_df['TreatmentClass'], categories=[classA, classB], ordered=True)
                
                accepted_count = 0
                rejected_count = 0
                
                for z_col in z_cols:
                    safe_col = z_col.replace('-', '_')
                    temp_df = pair_df.rename(columns={z_col: safe_col})
                    temp_df = temp_df.dropna(subset=[safe_col, 'TreatmentClass'])
                    
                    if len(temp_df['TreatmentClass'].unique()) < 2:
                        continue
                        
                    try:
                        formula = f"{safe_col} ~ C(TreatmentClass)"
                        md = smf.ols(formula, temp_df)
                        mdf = md.fit()
                        
                        term = f"C(TreatmentClass)[T.{classB}]"
                        if term in mdf.pvalues:
                            p_value = mdf.pvalues[term]
                            coef = mdf.params[term]
                        else:
                            treatment_terms = [k for k in mdf.pvalues.index if k != 'Intercept']
                            if treatment_terms:
                                term = treatment_terms[0]
                                p_value = mdf.pvalues[term]
                                coef = mdf.params[term]
                            else:
                                p_value = np.nan
                                coef = np.nan
                                
                        is_sig = not pd.isna(p_value) and p_value < 0.05
                        sig = "Sim" if is_sig else "Não"
                        if is_sig:
                            accepted_count += 1
                        else:
                            rejected_count += 1
                            
                        pairwise_results.append({
                            'Classe_A': classA,
                            'Classe_B': classB,
                            'Métrica Z-Score': z_col,
                            'Efeito (Coef)': coef,
                            'P-Value': p_value,
                            'Significativo?': sig
                        })
                    except Exception as e:
                        gui_log_callback(f"Aviso no cálculo OLS de {z_col} para {classA} vs {classB}: {e}")
                
                pair_summaries.append({
                    'Par': f"{classA} vs {classB}",
                    'Classe_A': classA,
                    'Classe_B': classB,
                    'Características Significativas (Aceito)': accepted_count,
                    'Características Não Significativas (Rejeitado)': rejected_count
                })
            
            # Ordena os resumos pelo número de características significativas (Aceito) decrescente
            pair_summaries_df = pd.DataFrame(pair_summaries)
            pair_summaries_df = pair_summaries_df.sort_values(by='Características Significativas (Aceito)', ascending=False)
            
            # Identifica o par com maior diferença
            if not pair_summaries_df.empty:
                most_diff_row = pair_summaries_df.iloc[0]
                most_diff_pair = most_diff_row['Par']
                most_diff_count = most_diff_row['Características Significativas (Aceito)']
                highlight_html = f"""
                <div class="highlight-box">
                    <h3>Destaque de Divergência Biológica</h3>
                    <p>O par de classes com <b>maior diferença geral</b> é <b>{most_diff_pair}</b>, apresentando <b>{most_diff_count}</b> características normalizadas com diferença estatisticamente significativa (p < 0.05).</p>
                </div>
                """
            
            # Salvar resultados consolidados
            pairwise_df = pd.DataFrame(pairwise_results)
            pairwise_df.to_csv(results_path / "pairwise_comparisons.csv", index=False)
            pair_summaries_df.to_csv(results_path / "pairwise_summary.csv", index=False)
            
            # Construir HTML das tabelas OLS
            ols_html = "<h3>Resumo de Comparação Pareada (Ranking de Divergência)</h3>"
            ols_html += "<table><thead><tr><th>Par de Classes</th><th>Features Aceitas (Significativo, p<0.05)</th><th>Features Rejeitadas (Não Significativo)</th></tr></thead><tbody>"
            for _, row in pair_summaries_df.iterrows():
                ols_html += f"<tr><td><b>{row['Par']}</b></td><td><span class='badge badge-success'>{row['Características Significativas (Aceito)']}</span></td><td><span class='badge badge-neutral'>{row['Características Não Significativas (Rejeitado)']}</span></td></tr>"
            ols_html += "</tbody></table>"
            
            ols_html += "<h3>Tabelas de Regressão OLS Detalhadas por Par</h3>"
            for _, row_summary in pair_summaries_df.iterrows():
                p_name = row_summary['Par']
                cA = row_summary['Classe_A']
                cB = row_summary['Classe_B']
                
                # Filtrar resultados desse par
                res_filtered = [r for r in pairwise_results if r['Classe_A'] == cA and r['Classe_B'] == cB]
                if not res_filtered:
                    continue
                
                # Gerar tabela HTML
                table_rows = ""
                for r in res_filtered:
                    p_val = r['P-Value']
                    p_str = "" if pd.isna(p_val) else (f"{p_val:.4e}" if p_val < 0.0001 else f"{p_val:.4f}")
                    coef_str = "" if pd.isna(r['Efeito (Coef)']) else f"{r['Efeito (Coef)']:.4f}"
                    badge = f"<span class='badge badge-success'>Sim</span>" if r['Significativo?'] == "Sim" else f"<span class='badge badge-neutral'>Não</span>"
                    url = _get_metric_url(r['Métrica Z-Score'])
                    metric_link = f"<a href='{url}' target='_blank' class='info-link'><b>{r['Métrica Z-Score']}</b> (info)</a>"
                    table_rows += f"<tr><td>{metric_link}</td><td>{coef_str}</td><td>{p_str}</td><td>{badge}</td></tr>"
                
                pair_table = f"""
                <table>
                    <thead>
                        <tr>
                            <th>Métrica Z-Score (Definição)</th>
                            <th>Efeito (Coeficiente)</th>
                            <th>P-Value</th>
                            <th>Significativo? (p < 0.05)</th>
                        </tr>
                    </thead>
                    <tbody>
                        {table_rows}
                    </tbody>
                </table>
                """
                
                ols_html += f"""
                <details>
                    <summary>{p_name} — ({row_summary['Características Significativas (Aceito)']} features significativas)</summary>
                    <div class="details-content">
                        <p>Modelo OLS ajustado: <code>Feature_Z ~ C(TreatmentClass)</code> restringido aos dados de <b>{cA}</b> e <b>{cB}</b> (referência: <b>{cA}</b>).</p>
                        {pair_table}
                    </div>
                </details>
                """
        else:
            ols_html = "<p><i>Dados insuficientes ou colunas Z-Score não encontradas para regressão pareada.</i></p>"
            
    except ImportError:
        ols_html = "<p><i>Biblioteca 'statsmodels' não instalada. Análise OLS ignorada.</i></p>"
    except Exception as e:
        ols_html = f"<p><i>Erro no cálculo OLS: {e}</i></p>"

    # --- ANÁLISES ESTATÍSTICAS E MACHINE LEARNING (PCA, CORRELAÇÃO, IMPORTÂNCIA, BOXPLOTS) ---
    features_json_data = {}
    try:
        gui_log_callback("Processando análises multivariadas e ML para features_data.json...")
        import json
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
        from sklearn.ensemble import RandomForestClassifier

        master_df = pd.concat(master_df_list, ignore_index=True)
        raw_cols = [c for c in numeric_cols if not c.endswith('_Z')]

        # 1. Séries temporais limpas (agrupadas por minuto para calcular a média dos replicates)
        time_series_data = {}
        for c_name, c_df in all_data.items():
            num_df = c_df.select_dtypes(include='number')
            if 'minute' not in num_df.columns and 'minute' in c_df.columns:
                num_df['minute'] = c_df['minute']
            c_df_grouped = num_df.groupby('minute', as_index=False).mean()
            c_df_grouped = c_df_grouped.sort_values('minute')
            c_df_clean = c_df_grouped.replace({np.nan: None})
            time_series_data[c_name] = c_df_clean.to_dict(orient='records')

        # 2. Correlação de Pearson (métricas brutas)
        corr_df = master_df[raw_cols].dropna(how='all').fillna(0)
        if not corr_df.empty:
            corr_matrix = corr_df.corr(method='pearson').fillna(0)
            correlation_data = {
                "metrics": raw_cols,
                "values": corr_matrix.values.tolist()
            }
        else:
            correlation_data = {"metrics": raw_cols, "values": [[0]*len(raw_cols)]*len(raw_cols)}

        # 3. PCA (2 Componentes)
        pca_data = {}
        try:
            pca_df = master_df.copy()
            pca_df[raw_cols] = pca_df[raw_cols].fillna(pca_df[raw_cols].mean()).fillna(0)
            scaler = StandardScaler()
            scaled_features = scaler.fit_transform(pca_df[raw_cols])
            pca = PCA(n_components=2)
            pca_coords = pca.fit_transform(scaled_features)
            
            pca_df['PC1'] = pca_coords[:, 0]
            pca_df['PC2'] = pca_coords[:, 1]
            
            for c_name in classes:
                c_pca = pca_df[pca_df['TreatmentClass'] == c_name]
                c_pca_clean = c_pca[['PC1', 'PC2', 'minute', 'source_file']].replace({np.nan: None})
                pca_data[c_name] = c_pca_clean.to_dict(orient='records')
        except Exception as e_pca:
            gui_log_callback(f"Aviso no cálculo do PCA: {e_pca}")
            for c_name in classes:
                pca_data[c_name] = []

        # 4. Importância Global de Características (Random Forest Classifier)
        importance_data = []
        confusion_data = {}
        decision_tree_data = {}
        pairwise_acc_data = {}

        # Helper to serialize decision tree structure
        def export_tree_structure(decision_tree, feature_names, class_names):
            tree_ = decision_tree.tree_
            def recurse(node):
                if tree_.feature[node] != -2:  # not a leaf
                    name = feature_names[tree_.feature[node]]
                    threshold = float(tree_.threshold[node])
                    left_child = recurse(tree_.children_left[node])
                    right_child = recurse(tree_.children_right[node])
                    return {
                        "is_leaf": False,
                        "feature": name,
                        "threshold": threshold,
                        "left": left_child,
                        "right": right_child,
                        "samples": int(tree_.n_node_samples[node])
                    }
                else:
                    # leaf node
                    val = tree_.value[node][0]
                    max_idx = int(np.argmax(val))
                    class_name = str(class_names[max_idx])
                    dist = {str(c): float(v) for c, v in zip(class_names, val)}
                    return {
                        "is_leaf": True,
                        "class": class_name,
                        "samples": int(tree_.n_node_samples[node]),
                        "distribution": dist
                    }
            return recurse(0)

        try:
            rf_df = master_df.copy()
            rf_df[raw_cols] = rf_df[raw_cols].fillna(rf_df[raw_cols].mean()).fillna(0)
            X_rf = rf_df[raw_cols]
            y_rf = rf_df['TreatmentClass']
            
            if len(y_rf.unique()) >= 2:
                rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
                rf.fit(X_rf, y_rf)
                importances = rf.feature_importances_
                
                for col, imp in zip(raw_cols, importances):
                    importance_data.append({
                        "feature": col,
                        "importance": float(imp)
                    })
                importance_data = sorted(importance_data, key=lambda x: x['importance'], reverse=True)

                # Accuracy / Confusion Matrix
                try:
                    from sklearn.model_selection import train_test_split
                    from sklearn.metrics import confusion_matrix, accuracy_score
                    
                    class_counts = y_rf.value_counts()
                    can_stratify = (class_counts >= 2).all() and len(y_rf) >= 2 * len(class_counts)
                    
                    if can_stratify:
                        X_train, X_test, y_train, y_test = train_test_split(
                            X_rf, y_rf, test_size=0.3, random_state=42, stratify=y_rf
                        )
                    else:
                        X_train, X_test, y_train, y_test = train_test_split(
                            X_rf, y_rf, test_size=0.3, random_state=42
                        )
                    
                    rf_eval = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
                    rf_eval.fit(X_train, y_train)
                    y_pred = rf_eval.predict(X_test)
                    
                    cm_classes = sorted(list(y_rf.unique()))
                    cm = confusion_matrix(y_test, y_pred, labels=cm_classes)
                    acc = float(accuracy_score(y_test, y_pred))
                    
                    confusion_data = {
                        "classes": cm_classes,
                        "matrix": cm.tolist(),
                        "accuracy": acc
                    }
                except Exception as e_cm:
                    gui_log_callback(f"Aviso no cálculo da matriz de confusão: {e_cm}")
                    try:
                        from sklearn.metrics import confusion_matrix, accuracy_score
                        rf_eval = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
                        rf_eval.fit(X_rf, y_rf)
                        y_pred = rf_eval.predict(X_rf)
                        cm_classes = sorted(list(y_rf.unique()))
                        cm = confusion_matrix(y_rf, y_pred, labels=cm_classes)
                        acc = float(accuracy_score(y_rf, y_pred))
                        confusion_data = {
                            "classes": cm_classes,
                            "matrix": cm.tolist(),
                            "accuracy": acc
                        }
                    except Exception:
                        confusion_data = {"classes": [], "matrix": [], "accuracy": 0.0}

                # Interactive Decision Tree
                try:
                    from sklearn.tree import DecisionTreeClassifier
                    dt = DecisionTreeClassifier(max_depth=10, random_state=42)
                    dt.fit(X_rf, y_rf)
                    decision_tree_data = export_tree_structure(dt, list(X_rf.columns), list(dt.classes_))
                except Exception as e_dt:
                    gui_log_callback(f"Aviso no cálculo da árvore de decisão: {e_dt}")

                # Pairwise Accuracy Matrix
                try:
                    from sklearn.model_selection import train_test_split
                    from sklearn.metrics import accuracy_score
                    unique_classes = sorted(list(y_rf.unique()))
                    matrix_size = len(unique_classes)
                    acc_matrix = [[1.0] * matrix_size for _ in range(matrix_size)]
                    
                    for i in range(matrix_size):
                        for j in range(i + 1, matrix_size):
                            c1 = unique_classes[i]
                            c2 = unique_classes[j]
                            
                            mask = y_rf.isin([c1, c2])
                            X_pair = X_rf[mask]
                            y_pair = y_rf[mask]
                            
                            if len(y_pair.unique()) == 2:
                                class_counts_pair = y_pair.value_counts()
                                can_stratify_pair = (class_counts_pair >= 2).all() and len(y_pair) >= 4
                                
                                try:
                                    if can_stratify_pair:
                                        X_tr, X_te, y_tr, y_te = train_test_split(
                                            X_pair, y_pair, test_size=0.3, random_state=42, stratify=y_pair
                                        )
                                    else:
                                        X_tr, X_te, y_tr, y_te = train_test_split(
                                            X_pair, y_pair, test_size=0.3, random_state=42
                                        )
                                    rf_pair = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42)
                                    rf_pair.fit(X_tr, y_tr)
                                    y_pred_pair = rf_pair.predict(X_te)
                                    pair_acc = float(accuracy_score(y_te, y_pred_pair))
                                except Exception:
                                    rf_pair = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42)
                                    rf_pair.fit(X_pair, y_pair)
                                    y_pred_pair = rf_pair.predict(X_pair)
                                    pair_acc = float(accuracy_score(y_pair, y_pred_pair))
                            else:
                                pair_acc = 0.5
                                
                            acc_matrix[i][j] = pair_acc
                            acc_matrix[j][i] = pair_acc
                            
                    pairwise_acc_data = {
                        "classes": unique_classes,
                        "matrix": acc_matrix
                    }

                    # Overwrite highlight_html with ML-based pairwise accuracy highlight
                    if matrix_size >= 2:
                        highest_acc = -1.0
                        most_divergent_pair = ""
                        for i in range(matrix_size):
                            for j in range(i + 1, matrix_size):
                                acc = acc_matrix[i][j]
                                if acc > highest_acc:
                                    highest_acc = acc
                                    most_divergent_pair = f"{unique_classes[i]} vs {unique_classes[j]}"
                        
                        if most_divergent_pair:
                            highlight_html = f"""
                            <div class="highlight-box">
                                <h3>Destaque de Divergência Biológica (Machine Learning)</h3>
                                <p>O par de classes com <b>maior diferença geral</b> é <b>{most_divergent_pair}</b>, apresentando uma acurácia de classificação binária de <b>{highest_acc * 100:.1f}%</b> no modelo Random Forest, indicando que seus biomarkers fisiológicos são altamente distinguíveis.</p>
                            </div>
                            """
                except Exception as e_pw:
                    gui_log_callback(f"Aviso no cálculo de acurácia pareada: {e_pw}")
                    pairwise_acc_data = {
                        "classes": [],
                        "matrix": []
                    }
            else:
                for col in raw_cols:
                    importance_data.append({"feature": col, "importance": 1.0 / len(raw_cols)})
        except Exception as e_rf:
            gui_log_callback(f"Aviso no cálculo de ML: {e_rf}")
            for col in raw_cols:
                importance_data.append({"feature": col, "importance": 1.0 / len(raw_cols)})

        # 5. Estatísticas de Boxplot
        boxplot_data = {}
        all_cols = list(raw_cols) + [c for c in numeric_cols if c.endswith('_Z')]
        for col in all_cols:
            boxplot_data[col] = {}
            for c_name, c_df in all_data.items():
                if col in c_df.columns:
                    col_vals = c_df[col].dropna().values
                    if len(col_vals) > 0:
                        boxplot_data[col][c_name] = {
                            "min": float(np.min(col_vals)),
                            "q1": float(np.percentile(col_vals, 25)),
                            "median": float(np.median(col_vals)),
                            "q3": float(np.percentile(col_vals, 75)),
                            "max": float(np.max(col_vals))
                        }
                    else:
                        boxplot_data[col][c_name] = {"min": 0, "q1": 0, "median": 0, "q3": 0, "max": 0}
                else:
                    boxplot_data[col][c_name] = {"min": 0, "q1": 0, "median": 0, "q3": 0, "max": 0}

        # Consolidar JSON
        features_json_data = {
            "time_series": time_series_data,
            "correlation": correlation_data,
            "pca": pca_data,
            "importance": importance_data,
            "boxplot": boxplot_data,
            "confusion_matrix": confusion_data,
            "decision_tree": decision_tree_data,
            "pairwise_accuracy": pairwise_acc_data
        }

        # Salvar features_data.json
        json_out_path = results_path / "features_data.json"
        with open(json_out_path, "w", encoding='utf-8') as jf:
            json.dump(features_json_data, jf, indent=2)
        gui_log_callback(f"Arquivo features_data.json salvo em: {json_out_path}")

        # Injeta os dados diretamente para funcionamento 100% offline
        json_data_str = json.dumps(features_json_data, ensure_ascii=False)

        # --- TIME-SERIES CLUSTER-BASED PERMUTATION TEST ---
        gui_log_callback("Processando análise de permutação baseada em clusters (Time-Series)...")
        try:
            # Obter a grade comum de minutos
            all_mins = []
            for c_name, c_df in all_data.items():
                if 'minute' in c_df.columns:
                    all_mins.extend(c_df['minute'].dropna().unique())
            all_mins = sorted(list(set(all_mins)))
            
            if not all_mins:
                all_mins = list(range(100))
                
            # Limita a grade a no máximo 500 pontos para manter a performance
            if len(all_mins) > 500:
                target_grid = np.linspace(all_mins[0], all_mins[-1], 500)
            else:
                target_grid = np.array(all_mins)
                
            # Determina min_replicates dinamicamente com base nas amostras disponíveis
            min_reps_across_classes = 9999
            for c_name, c_df in all_data.items():
                if 'source_file' in c_df.columns:
                    n_reps = len(c_df['source_file'].unique())
                    if n_reps < min_reps_across_classes:
                        min_reps_across_classes = n_reps
            min_reps = max(2, min(3, min_reps_across_classes))
            
            cluster_test_obj = TimeSeriesClusterTest(n_permutations=1000, alpha_local=0.05, min_replicates=min_reps)
            
            classes = sorted(all_data.keys())
            
            # Executa o teste de permutação para todas as colunas
            for col in all_cols:
                groups_list = []
                valid_classes = []
                
                for c_name in classes:
                    c_df = all_data[c_name]
                    if 'source_file' not in c_df.columns or col not in c_df.columns:
                        continue
                    
                    class_series = []
                    for rep in c_df['source_file'].unique():
                        rep_df = c_df[c_df['source_file'] == rep].sort_values('minute')
                        x = rep_df['minute'].values
                        y = rep_df[col].values
                        
                        mask = ~np.isnan(y) & ~np.isnan(x)
                        if np.sum(mask) < 2:
                            continue
                            
                        f_interp = scipy.interpolate.interp1d(
                            x[mask], y[mask],
                            kind='linear',
                            bounds_error=False,
                            fill_value=np.nan
                        )
                        resampled = f_interp(target_grid)
                        class_series.append(resampled)
                        
                    if len(class_series) >= min_reps:
                        groups_list.append(np.array(class_series))
                        valid_classes.append(c_name)
                
                if len(groups_list) >= 2:
                    clusters = cluster_test_obj.run_permutation_test(groups_list, target_grid)
                    passed = any(c['p_corrected'] < 0.05 for c in clusters)
                    
                    cluster_results_data[col] = {
                        'feature': col,
                        'clusters': clusters,
                        'passed': passed,
                        'valid_classes': valid_classes
                    }
        except Exception as e_cluster:
            gui_log_callback(f"Erro no cálculo do teste de permutação por clusters: {e_cluster}")
            import traceback
            gui_log_callback(traceback.format_exc())

    except Exception as e:
        gui_log_callback(f"Erro geral nas análises avançadas: {e}")

    # --- LÓGICA DE INTERPRETAÇÃO SIMPLES (SEM LLM) ---
    interpretations = []
    for col in summary_df.columns:
        try:
            best_class = summary_df[col].idxmax()
            max_val = summary_df[col].max()
            interpretations.append(f"O valor médio mais alto para a característica <b>{col}</b> foi <b>{max_val:.4f}</b> para a classe <b>{best_class}</b>.")
        except Exception:
            pass

    interpretations_html = ""
    if interpretations:
        list_items = "".join([f"<li>{item}</li>" for item in interpretations])
        interpretations_html = f"""
        <div class="section interpretations">
            <h2>Interpretação das Características (Resumo Estatístico)</h2>
            <ul>
                {list_items}
            </ul>
        </div>
        """

    gui_log_callback("Construindo HTML final...")


    # Formatar tabelas estatísticas como tabelas HTML responsivas
    summary_table_html = summary_df_transposed.to_html(border=0, classes='table summary-table', float_format=lambda x: f"{x:.4f}")
    comparison_table_html = comparison_df_transposed.to_html(border=0, classes='table comparison-table', float_format=lambda x: f"{x:.4f}")

    # Criar dicionário consolidado para o JSON
    report_data = {
        "experiment_description": str(experiment_description),
        "highlight_html": highlight_html,
        "interpretations_html": interpretations_html,
        "summary_table_html": summary_table_html,
        "comparison_table_html": comparison_table_html,
        "ols_html": ols_html,
        "charts": features_json_data,
        "cluster_analysis": cluster_results_data
    }

    report_json_path = results_path / "report_data.json"
    with open(report_json_path, "w", encoding='utf-8') as f:
        import json
        json.dump(report_data, f, indent=2)

    gui_log_callback(f"Dados do relatório salvos em: {report_json_path}")
    return report_json_path
