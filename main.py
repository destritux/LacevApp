import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
from ttkthemes import ThemedTk
from PIL import Image, ImageTk
import os
import threading
import queue
from pathlib import Path
import shutil
import warnings
from sklearn.exceptions import UndefinedMetricWarning
import traceback

# --- Suprimir avisos ---
warnings.filterwarnings('ignore', category=UserWarning, message='.*pkg_resources is deprecated.*')
warnings.filterwarnings('ignore', category=UndefinedMetricWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning, module='nolds')

# Importar lógica
import processing_logic
import plotting_logic
import report_logic
import platform

def resource_path(relative_path):
    """Retorna o caminho absoluto do recurso, compatível com PyInstaller e desenvolvimento."""
    import sys
    if hasattr(sys, '_MEIPASS'):
        return Path(sys._MEIPASS) / relative_path
    return Path(__file__).parent / relative_path


class ExperimentDescriptionDialog(tk.Toplevel):
    def __init__(self, parent):
        super().__init__(parent)
        self.transient(parent)
        self.title("Descrição do Experimento")
        self.geometry("500x300")
        self.protocol("WM_DELETE_WINDOW", self._on_cancel)
        self.description = None

        main_frame = ttk.Frame(self, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        ttk.Label(
            main_frame,
            text="Por favor, descreva o experimento (objetivos, métodos, o que as classes representam, etc.):"
        ).pack(pady=(0, 5), anchor="w")

        self.text_widget = scrolledtext.ScrolledText(main_frame, wrap=tk.WORD, height=10)
        self.text_widget.pack(fill=tk.BOTH, expand=True, pady=5)
        self.text_widget.focus_set()

        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=(10, 0))

        ttk.Button(button_frame, text="Gerar Relatório", command=self._on_ok).pack(side=tk.RIGHT, padx=5)
        ttk.Button(button_frame, text="Cancelar", command=self._on_cancel).pack(side=tk.RIGHT)

        self.grab_set()

    def _on_ok(self, event=None):
        self.description = self.text_widget.get("1.0", tk.END).strip()
        if not self.description:
            messagebox.showwarning(
                "Aviso",
                "A descrição está vazia. Por favor, forneça alguns detalhes sobre o experimento.",
                parent=self
            )
            return
        self.destroy()

    def _on_cancel(self):
        self.description = None
        self.destroy()


class LacevApp(ThemedTk):
    def __init__(self):
        super().__init__()
        try:
            self.set_theme("equilux")
        except tk.TclError:
            pass

        self.title("LacevApp - Plant Electrophysiology Processor")
        self.geometry("900x700")

        # Variáveis de estado
        self.base_path = tk.StringVar()
        self.total_time_minutes = tk.StringVar(value="1440")
        self.window_minutes = tk.StringVar(value="1")
        self.powerline_loc = tk.StringVar(value="America (60Hz)")
        
        # Novas Variáveis
        self.topology_var = tk.StringVar(value="Single-Channel")
        self.baseline_minutes = tk.StringVar(value="10")

        self.theme_dark_var = tk.BooleanVar(value=True)
        self.theme_light_var = tk.BooleanVar(value=True)

        self.feature_groups = ['ApEn', 'DFA', 'Lyapunov', 'Raw_Stats',
                               'FFT_Stats', 'PSD_Stats', 'Bandpower', 'Z_Scores', 'ISO']
        self.feature_vars = {group: tk.BooleanVar(value=True) for group in self.feature_groups}
        
        self.stop_event = threading.Event()

        try:
            ico_path = resource_path("assets/lacev-App.ico")
            png_path = resource_path("assets/lacev-App.png")
            if ico_path.exists():
                if platform.system() == 'Windows':
                    self.iconbitmap(ico_path)
                elif png_path.exists():
                    icon_img = tk.PhotoImage(file=str(png_path))
                    self.iconphoto(True, icon_img)
        except Exception:
            pass

        self.log_queue = queue.Queue()
        self._create_widgets()
        self.after(100, self._process_log_queue)

    def _create_widgets(self):
        container = ttk.Frame(self)
        container.pack(fill=tk.BOTH, expand=True)

        # Canvas com fundo preto para contraste premium
        canvas = tk.Canvas(container, bg="black", highlightthickness=0)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        scrollbar = ttk.Scrollbar(container, orient="vertical", command=canvas.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.configure(yscrollcommand=scrollbar.set)

        # Adicionar padding interno de 15px para respirar e dar toque moderno
        self.main_frame = ttk.Frame(canvas, padding="15")
        
        # Criar a janela do canvas e guardar o ID
        canvas_window = canvas.create_window((0, 0), window=self.main_frame, anchor="nw")

        # Ajusta dinamicamente a largura da janela interna e o tamanho do logotipo responsivamente
        def _on_canvas_configure(event):
            canvas.itemconfig(canvas_window, width=event.width)
            
            # Redimensiona o logotipo para ocupar todo o campo na horizontal (mantendo proporção)
            if hasattr(self, 'logo_img') and hasattr(self, 'logo_label'):
                avail_width = event.width - 30 # desconta o padding de 15px em cada lado
                if avail_width > 100:
                    orig_w, orig_h = self.logo_img.size
                    new_h = int(avail_width * (orig_h / orig_w))
                    
                    resized_img = self.logo_img.resize((avail_width, new_h), Image.Resampling.LANCZOS)
                    self.logo_photo = ImageTk.PhotoImage(resized_img)
                    self.logo_label.configure(image=self.logo_photo)

        canvas.bind('<Configure>', _on_canvas_configure)

        self.main_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        def _on_mousewheel(event):
            if event.num == 4:
                canvas.yview_scroll(-1, "units")
            elif event.num == 5:
                canvas.yview_scroll(1, "units")
            elif event.delta:
                canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        canvas.bind_all("<Button-4>", _on_mousewheel)
        canvas.bind_all("<Button-5>", _on_mousewheel)
        canvas.bind_all("<MouseWheel>", _on_mousewheel)

        try:
            logo_path = resource_path("assets/lacev-App.png")
            self.logo_img = Image.open(logo_path)
            self.logo_label = ttk.Label(self.main_frame)
            self.logo_label.pack(pady=5, fill=tk.X, expand=True)
        except Exception:
            title_label = ttk.Label(self.main_frame, text="LacevApp", font=("Helvetica", 24, "bold"))
            title_label.pack(pady=10)

        # Settings
        settings_frame = ttk.LabelFrame(self.main_frame, text="1. Settings", padding="10")
        settings_frame.pack(fill=tk.X, expand=True, pady=10)

        # Habilitar colunas auto-expansíveis para layout profissional e moderno
        settings_frame.columnconfigure(1, weight=1)
        settings_frame.columnconfigure(3, weight=1)
        settings_frame.columnconfigure(5, weight=1)

        ttk.Button(settings_frame, text="Open Folder", command=self._select_folder).grid(
            row=0, column=0, padx=5, pady=5, sticky="ew"
        )
        ttk.Entry(settings_frame, textvariable=self.base_path, state="readonly").grid(
            row=0, column=1, columnspan=5, padx=5, pady=5, sticky="ew"
        )

        ttk.Label(settings_frame, text="Tempo Total (min):").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        ttk.Entry(settings_frame, textvariable=self.total_time_minutes, width=10).grid(row=1, column=1, padx=5, pady=5, sticky="ew")

        ttk.Label(settings_frame, text="Janela (min):").grid(row=1, column=2, padx=5, pady=5, sticky="w")
        ttk.Entry(settings_frame, textvariable=self.window_minutes, width=10).grid(row=1, column=3, padx=5, pady=5, sticky="ew")
        
        ttk.Label(settings_frame, text="Baseline Z-Score (min):").grid(row=1, column=4, padx=5, pady=5, sticky="w")
        ttk.Entry(settings_frame, textvariable=self.baseline_minutes, width=10).grid(row=1, column=5, padx=5, pady=5, sticky="ew")

        ttk.Label(settings_frame, text="Filtro Notch:").grid(row=2, column=0, padx=5, pady=5, sticky="w")
        ttk.Combobox(settings_frame, textvariable=self.powerline_loc,
                     values=["America (60Hz)", "Europe (50Hz)"], state="readonly", width=15).grid(
            row=2, column=1, padx=5, pady=5, sticky="ew"
        )

        ttk.Label(settings_frame, text="Topologia:").grid(row=2, column=2, padx=5, pady=5, sticky="w")
        ttk.Combobox(settings_frame, textvariable=self.topology_var,
                     values=["Single-Channel", "Multi-Channel"], state="readonly", width=15).grid(
            row=2, column=3, padx=5, pady=5, sticky="ew"
        )

        # Actions
        actions_frame = ttk.LabelFrame(self.main_frame, text="2. Actions", padding="10")
        actions_frame.pack(fill=tk.X, expand=True, pady=10)

        self.btn_generate_csv = ttk.Button(actions_frame, text="Generate TDAF-CSV", command=self._start_processing)
        self.btn_generate_csv.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=5)

        self.btn_generate_report = ttk.Button(actions_frame, text="Generate Report (Statsmodels)", command=self._start_report_generation)
        self.btn_generate_report.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=5)

        self.btn_stop = ttk.Button(actions_frame, text="⏹ Stop Process", command=self._stop_action, state=tk.DISABLED)
        self.btn_stop.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=5)

        # Plotting
        plotting_frame = ttk.LabelFrame(self.main_frame, text="3. Plotting", padding="10")
        plotting_frame.pack(fill=tk.X, expand=True, pady=10)

        theme_frame = ttk.LabelFrame(plotting_frame, text="Temas dos Gráficos", padding=5)
        theme_frame.pack(fill=tk.X, expand=True, pady=5)
        ttk.Checkbutton(theme_frame, text="Escuro", variable=self.theme_dark_var).pack(side=tk.LEFT, padx=10)
        ttk.Checkbutton(theme_frame, text="Claro", variable=self.theme_light_var).pack(side=tk.LEFT, padx=10)

        features_frame = ttk.LabelFrame(plotting_frame, text="Características a Plotar", padding=5)
        features_frame.pack(fill=tk.X, expand=True, pady=5)
        
        # Grid para as features não ficarem muito longas numa linha
        for i, (group, var) in enumerate(self.feature_vars.items()):
            ttk.Checkbutton(features_frame, text=group, variable=var).grid(row=i//4, column=i%4, padx=10, pady=5, sticky="w")

        self.btn_generate_graphs = ttk.Button(plotting_frame, text="Generate Selected Graphs", command=self._start_plotting)
        self.btn_generate_graphs.pack(fill=tk.X, expand=True, pady=10, padx=5)

        # Feedback
        feedback_frame = ttk.LabelFrame(self.main_frame, text="4. Log & Progress", padding="10")
        feedback_frame.pack(fill=tk.BOTH, expand=True, pady=10)

        self.progress_bar = ttk.Progressbar(feedback_frame, orient="horizontal", mode="determinate")
        self.progress_bar.pack(fill=tk.X, expand=True, pady=5)

        log_frame = ttk.Frame(feedback_frame)
        log_frame.pack(fill=tk.BOTH, expand=True)
        self.log_text = tk.Text(log_frame, state="disabled", wrap="word", height=10, bg="#2b3035", fg="white")
        log_scrollbar = ttk.Scrollbar(log_frame, orient="vertical", command=self.log_text.yview)
        self.log_text.config(yscrollcommand=log_scrollbar.set)
        log_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    def _select_folder(self):
        path = filedialog.askdirectory()
        if path:
            raw_path = os.path.join(path, "raw")
            if not os.path.isdir(raw_path):
                messagebox.showerror("Erro", "A pasta selecionada deve conter uma subpasta 'raw'.")
                return
            self.base_path.set(path)
            self._log(f"Pasta base selecionada: {path}")

    def _log(self, message):
        self.log_queue.put(message)

    def _process_log_queue(self):
        while not self.log_queue.empty():
            try:
                message = self.log_queue.get_nowait()
                self.log_text.config(state="normal")
                self.log_text.insert(tk.END, str(message) + "\n")
                self.log_text.see(tk.END)
                self.log_text.config(state="disabled")
            except queue.Empty:
                pass
        self.after(100, self._process_log_queue)

    def _update_progress(self, value):
        self.progress_bar['value'] = value
        self.update_idletasks()
    
    def _clear_environment(self):
        self.log_text.config(state='normal')
        self.log_text.delete('1.0', tk.END)
        self.log_text.config(state='disabled')
        
        with self.log_queue.mutex:
            self.log_queue.queue.clear()
            
        self.progress_bar['value'] = 0
        
        if self.base_path.get():
            base = Path(self.base_path.get())
            for folder in ["filtered", "features", "graphics"]:
                dir_path = base / folder
                if dir_path.exists():
                    try:
                        shutil.rmtree(dir_path)
                    except Exception as e:
                        print(f"Erro ao limpar {folder}: {e}")
                        
        self._log("🧹 Ambiente e arquivos residuais limpos. Pronto para nova análise.")
    
    def _stop_action(self):
        self._log("⚠️ Solicitação de interrupção recebida. Parando processo e limpando ambiente...")
        self.stop_event.set()
        self.btn_stop.config(state=tk.DISABLED)

    def _validate_inputs(self):
        if not self.base_path.get():
            messagebox.showerror("Erro de Input", "Por favor, selecione uma pasta base.")
            return False
        try:
            float(self.total_time_minutes.get())
            float(self.window_minutes.get())
            float(self.baseline_minutes.get())
        except ValueError:
            messagebox.showerror("Erro de Input", "O Tempo Total, Janela e Baseline devem ser números válidos.")
            return False
        return True

    def _set_ui_state(self, state):
        status = tk.DISABLED if state == 'disabled' else tk.NORMAL
        self.btn_generate_csv.config(state=status)
        self.btn_generate_graphs.config(state=status)
        self.btn_generate_report.config(state=status)
        self.btn_stop.config(state=tk.NORMAL if state == 'disabled' else tk.DISABLED)

    def _start_processing(self):
        if not self._validate_inputs():
            return
        self._set_ui_state('disabled')
        self.stop_event.clear()
        self._update_progress(0)
        
        total_time = float(self.total_time_minutes.get())
        win_min = float(self.window_minutes.get())
        baseline_min = float(self.baseline_minutes.get())
        notch = 60 if "60Hz" in self.powerline_loc.get() else 50
        topology = self.topology_var.get()
        
        proc_thread = threading.Thread(
            target=self._processing_worker,
            args=(self.base_path.get(), total_time, win_min, baseline_min, notch, topology)
        )
        proc_thread.daemon = True
        proc_thread.start()

    def _processing_worker(self, base_path, total_time, win_min, baseline_min, notch, topology):
        try:
            processor = processing_logic.SignalProcessor(
                base_path=base_path, total_time_minutes=total_time, window_minutes=win_min,
                baseline_minutes=baseline_min, notch_freq=notch, topology=topology,
                gui_log_callback=self._log, stop_event=self.stop_event
            )
            processor.run_tdsf_extraction(progress_callback=self._update_progress)
            
            if not self.stop_event.is_set():
                self._log("\n✅ Processamento concluído com sucesso!")
        except Exception as e:
            self._log(f"\n❌ Ocorreu um erro: {e}")
            self._log(traceback.format_exc())
        finally:
            self._set_ui_state('normal')
            self._update_progress(0)
            if self.stop_event.is_set():
                self._clear_environment()

    def _start_plotting(self):
        if not self.base_path.get():
            messagebox.showerror("Erro de Input", "Por favor, selecione primeiro uma pasta base.")
            return

        selected_themes = []
        if self.theme_dark_var.get():
            selected_themes.append('Escuro')
        if self.theme_light_var.get():
            selected_themes.append('Claro')

        if not selected_themes:
            messagebox.showerror("Erro de Input", "Por favor, selecione pelo menos um tema para os gráficos.")
            return

        selected_features = [group for group, var in self.feature_vars.items() if var.get()]

        if not selected_features:
            messagebox.showerror("Erro de Input", "Por favor, selecione pelo menos uma característica para plotar.")
            return

        self._set_ui_state('disabled')
        self.stop_event.clear()
        plot_thread = threading.Thread(
            target=self._plotting_worker,
            args=(self.base_path.get(), selected_features, selected_themes)
        )
        plot_thread.daemon = True
        plot_thread.start()

    def _plotting_worker(self, base_path, selected_features, selected_themes):
        try:
            for feature_group in selected_features:
                if self.stop_event.is_set(): break
                self._log(f"--- A iniciar a Geração de Gráficos para {feature_group} ---")
                plotting_logic.plot_tdsf(base_path, feature_group, themes_to_plot=selected_themes,
                                         log_scale=False, gui_log_callback=self._log)
                if 'Stats' in feature_group or 'Bandpower' in feature_group:
                    if self.stop_event.is_set(): break
                    plotting_logic.plot_tdsf(base_path, feature_group, themes_to_plot=selected_themes,
                                             log_scale=True, gui_log_callback=self._log)
            if not self.stop_event.is_set():
                self._log(f"\n✅ Plotagem concluída com sucesso!")
        except Exception as e:
            self._log(f"\n❌ Ocorreu um erro durante a plotagem: {e}")
            self._log(traceback.format_exc())
        finally:
            self._set_ui_state('normal')
            if self.stop_event.is_set():
                self._clear_environment()

    def _start_report_generation(self):
        if not self.base_path.get():
            messagebox.showerror("Erro de Input", "Por favor, selecione primeiro uma pasta base.")
            return
        self._set_ui_state('disabled')
        report_thread = threading.Thread(
            target=self._report_worker,
            args=(self.base_path.get(), "Relatório de análise eletrofisiológica gerado automaticamente pelo LacevApp.")
        )
        report_thread.daemon = True
        report_thread.start()

    def _report_worker(self, base_path, experiment_description):
        try:
            self._log("--- A iniciar a Geração do Relatório ---")
            report_logic.generate_html_report(
                base_path=base_path,
                experiment_description=experiment_description,
                gui_log_callback=self._log
            )
            report_path = Path(base_path) / "results" / "report.html"
            self._log(f"\n✅ Relatório gerado com sucesso! Guardado em: {report_path}")
            try:
                os.startfile(report_path)
            except AttributeError:
                import subprocess
                import platform
                if platform.system() == 'Darwin':
                    subprocess.call(['open', report_path])
                else:
                    subprocess.call(['xdg-open', report_path])
        except Exception as e:
            self._log(f"\nOcorreu um erro durante a geração do relatório: {e}")
            self._log(traceback.format_exc())
        finally:
            self._set_ui_state('normal')


if __name__ == "__main__":
    try:
        app = LacevApp()
        app.mainloop()
    except Exception as e:
        print("--- OCORREU UM ERRO CRÍTICO AO INICIAR O APLICATIVO ---")
        traceback.print_exc()
        print("---------------------------------------------------------")
        input("Pressione ENTER para fechar a janela...")