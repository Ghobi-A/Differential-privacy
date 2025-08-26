"""Simple Tkinter GUI for applying differential privacy mechanisms to CSV files."""

import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd

from mechanisms import (
    add_laplace_noise,
    add_gaussian_noise,
    add_exponential_noise,
    add_geometric_noise,
    randomised_response,
)

MECHANISMS = [
    "Laplace",
    "Gaussian",
    "Exponential",
    "Geometric",
    "Randomised Response",
]


def main() -> None:
    """Launch the GUI."""
    root = tk.Tk()
    root.title("Differential Privacy GUI")

    # Variables
    input_var = tk.StringVar()
    output_var = tk.StringVar()
    mechanism_var = tk.StringVar(value=MECHANISMS[0])
    epsilon_var = tk.StringVar()
    delta_var = tk.StringVar()
    sensitivity_var = tk.StringVar()
    seed_var = tk.StringVar()

    def browse_input() -> None:
        path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if path:
            input_var.set(path)

    def browse_output() -> None:
        path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
        if path:
            output_var.set(path)

    def run() -> None:
        """Apply the selected mechanism and save the result."""
        inp = input_var.get()
        out = output_var.get()
        mech = mechanism_var.get()
        try:
            df = pd.read_csv(inp)
        except Exception as exc:  # pragma: no cover - GUI feedback
            messagebox.showerror("Error", f"Failed to read input file: {exc}")
            return

        numeric = df.select_dtypes(include="number")
        categorical = df.select_dtypes(exclude="number")

        try:
            eps = float(epsilon_var.get()) if epsilon_var.get() else 0.1
            delt = float(delta_var.get()) if delta_var.get() else 1e-5
            sens = float(sensitivity_var.get()) if sensitivity_var.get() else 1.0
            seed = int(seed_var.get()) if seed_var.get() else None

            if mech == "Laplace":
                df[numeric.columns] = add_laplace_noise(
                    numeric, epsilon=eps, sensitivity=sens, random_state=seed
                )
            elif mech == "Gaussian":
                df[numeric.columns] = add_gaussian_noise(
                    numeric,
                    epsilon=eps,
                    delta=delt,
                    sensitivity=sens,
                    random_state=seed,
                )
            elif mech == "Exponential":
                df[numeric.columns] = add_exponential_noise(
                    numeric, epsilon=eps, sensitivity=sens, random_state=seed
                )
            elif mech == "Geometric":
                df[numeric.columns] = add_geometric_noise(
                    numeric, epsilon=eps, random_state=seed
                )
            elif mech == "Randomised Response":
                for col in categorical.columns:
                    df[col] = randomised_response(categorical[col], random_state=seed)

            df.to_csv(out, index=False)
        except Exception as exc:  # pragma: no cover - GUI feedback
            messagebox.showerror("Error", str(exc))
            return

        messagebox.showinfo("Success", f"Saved output to {out}")

    # Layout
    tk.Label(root, text="Input CSV:").grid(row=0, column=0, sticky="e")
    tk.Entry(root, textvariable=input_var, width=40).grid(row=0, column=1)
    tk.Button(root, text="Browse", command=browse_input).grid(row=0, column=2)

    tk.Label(root, text="Output CSV:").grid(row=1, column=0, sticky="e")
    tk.Entry(root, textvariable=output_var, width=40).grid(row=1, column=1)
    tk.Button(root, text="Browse", command=browse_output).grid(row=1, column=2)

    tk.Label(root, text="Mechanism:").grid(row=2, column=0, sticky="e")
    tk.OptionMenu(root, mechanism_var, *MECHANISMS).grid(row=2, column=1, sticky="w")

    tk.Label(root, text="Epsilon:").grid(row=3, column=0, sticky="e")
    tk.Entry(root, textvariable=epsilon_var).grid(row=3, column=1, sticky="w")

    tk.Label(root, text="Delta:").grid(row=4, column=0, sticky="e")
    tk.Entry(root, textvariable=delta_var).grid(row=4, column=1, sticky="w")

    tk.Label(root, text="Sensitivity:").grid(row=5, column=0, sticky="e")
    tk.Entry(root, textvariable=sensitivity_var).grid(row=5, column=1, sticky="w")

    tk.Label(root, text="Random seed:").grid(row=6, column=0, sticky="e")
    tk.Entry(root, textvariable=seed_var).grid(row=6, column=1, sticky="w")

    tk.Button(root, text="Run", command=run).grid(row=7, column=1)

    root.mainloop()


if __name__ == "__main__":
    main()
