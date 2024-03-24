import tkinter as tk
from tkinter import ttk
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class UniqueCustomerSegmentationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Distinct Customer Segmentation App")

        self.configure_ui()

        # Load the dataset
        self.dataset = pd.read_csv("../PRODIGY_ML_2/Mall_Customers.csv")

    def configure_ui(self):
        self.root.configure(bg="#f0f0f0")

        self.main_frame = ttk.Frame(root, padding="20", style="UniqueMain.TFrame")
        self.main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.main_frame.columnconfigure(0, weight=1)
        self.main_frame.rowconfigure(0, weight=1)

        ttk.Style().configure("UniqueMain.TFrame", background="#f0f0f0")

        self.create_widgets()

    def create_widgets(self):
        ttk.Label(self.main_frame, text="Input Customer Information", font=("Helvetica", 16), style="UniqueTitle.TLabel").grid(row=0, column=0, columnspan=3, pady=10)
        ttk.Style().configure("UniqueTitle.TLabel", foreground="green", background="#f0f0f0")

        ttk.Label(self.main_frame, text="Gender:", style="UniqueSubTitle.TLabel").grid(row=1, column=0, sticky=tk.E, pady=5)
        ttk.Style().configure("UniqueSubTitle.TLabel", font=("Helvetica", 12), background="#f0f0f0")

        self.gender_var = tk.StringVar()
        male_radio = ttk.Radiobutton(self.main_frame, text="Male", variable=self.gender_var, value="Male")
        female_radio = ttk.Radiobutton(self.main_frame, text="Female", variable=self.gender_var, value="Female")

        male_radio.grid(row=1, column=1, pady=5)
        female_radio.grid(row=1, column=2, pady=5)

        ttk.Label(self.main_frame, text="Age:", style="UniqueSubTitle.TLabel").grid(row=2, column=0, sticky=tk.E, pady=5)
        self.age_entry = ttk.Entry(self.main_frame)
        self.age_entry.grid(row=2, column=1, pady=5)

        ttk.Label(self.main_frame, text="Annual Income (k$):", style="UniqueSubTitle.TLabel").grid(row=3, column=0, sticky=tk.E, pady=5)
        self.income_entry = ttk.Entry(self.main_frame)
        self.income_entry.grid(row=3, column=1, pady=5)

        ttk.Label(self.main_frame, text="Spending Score (1-100):", style="UniqueSubTitle.TLabel").grid(row=4, column=0, sticky=tk.E, pady=5)
        self.score_entry = ttk.Entry(self.main_frame)
        self.score_entry.grid(row=4, column=1, pady=5)

        segmentation_button = ttk.Button(self.main_frame, text="Analyze Segmentation", command=self.perform_segmentation)
        segmentation_button.grid(row=5, column=0, columnspan=3, pady=10)

        self.fig, self.ax = plt.subplots(figsize=(8, 6), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))

        self.cluster_label = ttk.Label(self.root, text="", font=("Helvetica", 10), wraplength=150)
        self.cluster_label.grid(row=0, column=2, padx=10, pady=10, sticky=tk.N)

    def perform_segmentation(self):
        gender = self.gender_var.get()
        age = int(self.age_entry.get())
        income = int(self.income_entry.get())
        score = int(self.score_entry.get())

        new_data = pd.DataFrame({'Age': [age], 'Annual Income (k$)': [income], 'Spending Score (1-100)': [score]})
        data_for_clustering = pd.concat([self.dataset[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']], new_data], ignore_index=True)

        kmeans = KMeans(n_clusters=5, random_state=42)
        data_for_clustering['Cluster'] = kmeans.fit_predict(data_for_clustering[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']])

        self.display_clusters(data_for_clustering)

    def display_clusters(self, data):
        self.ax.clear()

        unique_clusters = data['Cluster'].unique()
        cluster_names = {cluster: f'Cluster {cluster}' for cluster in unique_clusters}

        colors = ['red', 'blue', 'green', 'purple', 'orange']
        for cluster in unique_clusters:
            cluster_data = data[data['Cluster'] == cluster]
            self.ax.scatter(cluster_data['Annual Income (k$)'], cluster_data['Spending Score (1-100)'],
                            color=colors[cluster], label=cluster_names[cluster])

        self.ax.set_xlabel('Annual Income (k$)')
        self.ax.set_ylabel('Spending Score (1-100)')
        self.ax.legend()

        self.canvas.draw()

        cluster_label_text = "\n".join([f"{cluster_names[cluster]}" for cluster in unique_clusters])
        self.cluster_label.config(text=cluster_label_text)

if __name__ == "__main__":
    root = tk.Tk()
    app = UniqueCustomerSegmentationApp(root)
    root.mainloop()






