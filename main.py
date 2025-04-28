import customtkinter as ctk
from pywinstyles import apply_style
import pywinstyles as pws
from tkinter import messagebox
import sympy as sp
from sympy import lambdify, symbols
from sympy.parsing.sympy_parser import (
            parse_expr,
            standard_transformations,
            implicit_multiplication_application,
            convert_xor,
        )
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.textpath import TextPath
from matplotlib.patches import PathPatch
from matplotlib.font_manager import FontProperties
from matplotlib.backends.backend_tkagg import (
    NavigationToolbar2Tk,
    FigureCanvasTkAgg as FigureCanvas
)
from matplotlib.figure import Figure
import PIL
from PIL import Image, ImageTk
import io
import re


# Styling colors
main_color = 'white'
secondary_color = '#cccccc'
button_color = '#3B8ED0'
background_text_color = 'black'
button_text_color = "black"

# Text entry selection manager
current_focus = None

# Second button
second = False
def toggle_second():
    global second, button_color
    if second is True:
        second = False
        second_button.configure(text_color='black', fg_color=button_color)
        sine_button.configure(text='sin(x)', font=("Roboto", 13))
        cosine_button.configure(text='cos(x)', font=("Roboto", 13))
        tangent_button.configure(text='tan(x)', font=("Roboto", 13))
        antiderivative_button.configure(text="∫dx")
        e_button.configure(text='e^')
        nCr_button.configure(text="nCr")
        eval_button.configure(text="Enter (=)")
    elif second is False:
        second = True
        second_button.configure(text_color='gray', fg_color='lightgray')
        sine_button.configure(text='sin⁻¹(x)', font=("Roboto", 10))
        cosine_button.configure(text='cos⁻¹(x)', font=("Roboto", 10))
        tangent_button.configure(text='tan⁻¹(x)', font=("Roboto", 10))
        antiderivative_button.configure(text='∫ₐᵇ')
        e_button.configure(text='ln')
        nCr_button.configure(text="nPr")
        eval_button.configure(text="Enter (≈)")

#definition for nPr
def nPr(n, r):
    return sp.factorial(n)/sp.factorial(n-r)

# Memory Locations
mem1 = None
mem2 = None
mem3 = None
matrix_mem = None
matrix_mem2 = None
canvas_widget = None
toolbar = None

# popup reference for menus and windows
popup = None

# Define known constants mapping
KNOWN_CONSTANTS = {
    "oo": sp.oo,
    "infinity": sp.oo,
    "∞": sp.oo,
    "pi": sp.pi,
    "π": sp.pi,
    "e": sp.E,
    "E": sp.E,
    "i": sp.I,
    "I": sp.I,
    "sqrt": sp.sqrt,
    "nPr": nPr
}

# Closing protocol
def on_closing():
    # Destroy all widgets on close
    for widget in root.winfo_children():
        widget.destroy()
    root.quit()

'''
Character insert functions
'''
# Controls which entry buttons input into
def set_focus(entry):
    global current_focus
    current_focus = entry

# Insert character
def insert_char(char):
    if current_focus:
        current = current_focus.get()
        current_focus.delete(0, ctk.END)
        current_focus.insert(0, current + char)

# Backspace
def backspace():
    current = current_focus.get()
    current_focus.delete(0, ctk.END)
    current_focus.insert(0, current[:-1])

# Clear all
def clear():
    text_entry.delete(0, ctk.END)

# Insert Pi
def insert_pi():
    if current_focus:
        current_focus.insert("end", "π")

# Insert e or 2nd insert ln
def insert_e():
    if current_focus:
        if second is False:
            current_focus.insert("end", "e^")
        elif second is True:
            current_focus.insert("end", "ln(")

# Insert infinity
def insert_infinity():
    if current_focus:
        current_focus.insert("end", "∞")

def insert_radical():
    if current_focus:
        current_focus.insert("end", "√")

def insert_i():
    if current_focus:
        current_focus.insert("end", "i")

'''
LaTeX rendering function
'''

def render_latex(latex_expr, fontsize=20, dpi=300, padding=20, bg_alpha=0):

    # Font and text path
    font = FontProperties(size=fontsize)
    text_path = TextPath((0, 0), f"${latex_expr}$", prop=font, usetex=False)
    x0, y0 = text_path.vertices.min(axis=0)
    x1, y1 = text_path.vertices.max(axis=0)

    # Add padding
    width = (x1 - x0) + 2 * padding
    height = (y1 - y0) + 2 * padding

    # Create figure and axis
    fig = Figure(figsize=(width / dpi, height / dpi), dpi=dpi)
    fig.patch.set_alpha(bg_alpha)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(x0 - padding, x1 + padding)
    ax.set_ylim(y0 - padding, y1 + padding)
    ax.axis("off")

    # Add colored text
    patch = PathPatch(text_path, facecolor=background_text_color, lw=0)
    ax.add_patch(patch)

    # Render to buffer
    buf = io.BytesIO()
    canvas = FigureCanvas(fig)
    canvas.print_png(buf)
    buf.seek(0)

    # Open and convert to CTkImage
    image = Image.open(buf).convert("RGBA")
    return ctk.CTkImage(light_image=image, size=image.size)

'''
Preprocess input function
'''

def preprocess_input(text):
    replacements = {
        "∞": "oo",
        "π": "pi",
        "×": "*",
        "÷": "/",
        "^": "**",
        "nCr": "binomial"
    }
    for k, v in replacements.items():
        text = text.replace(k, v)

    # Regex: replace √expression with sqrt(expression)
    # Handles √2, √(3 + x), etc.
    text = re.sub(r"√\s*(\w+|\([^)]*\))", r"sqrt(\1)", text)

    return text

"""
Function to handle user friendly input
"""

def convert_to_sympy_value(expr_string):
    # Build the local_dict from known constants
    local_dict = {name: value for name, value in KNOWN_CONSTANTS.items()}

    # Parse the expression safely
    try:
        expr = parse_expr(expr_string,
                          transformations=standard_transformations + (implicit_multiplication_application,),
                          local_dict=local_dict)
        return expr
    except Exception as e:
        messagebox.showerror('Error Occurred', f"Parsing error: {e}")
        return None


'''
Graphing Function
'''


def plot_function():
    x = sp.Symbol('x')
    expr = text_entry.get()
    expr_str = preprocess_input(expr)  # This should replace √ with sqrt, and do other cleanup
    global toolbar

    try:
        functions = [f.strip() for f in expr_str.split(',')]

        x_vals = np.linspace(-10, 10, 800)
        fig = Figure(figsize=(6, 4), dpi=100)
        ax = fig.add_subplot(111)

        min_y = float('inf')
        max_y = float('-inf')

        for idx, func_str in enumerate(functions):
            expr = convert_to_sympy_value(func_str)
            func = sp.lambdify(x, expr, modules=["numpy"])

            try:
                y_vals = func(x_vals)

                # Filter invalid (nan, inf, complex) values
                filtered_x = []
                filtered_y = []
                for xi, yi in zip(x_vals, y_vals):
                    if isinstance(yi, complex):
                        continue
                    if np.isnan(yi) or np.isinf(yi):
                        continue
                    filtered_x.append(xi)
                    filtered_y.append(yi)

                if not filtered_x:
                    raise ValueError("Function has no valid real outputs in the given range.")

                min_y = min(min_y, min(filtered_y))
                max_y = max(max_y, max(filtered_y))

                ax.plot(filtered_x, filtered_y, label=f"f{x} = {func_str}", linewidth=2)

            except Exception as e:
                messagebox.showerror("Error in function",
                                     f"Could not evaluate the function: "f"{func_str}\nError: {e}")
                return

        margin = 0.1
        ax.set_xlim(-10, 10)
        ax.set_ylim(min_y - margin, max_y + margin)

        ax.grid(True)
        plot_title = '   '.join([f'f{i+1}(x) = {f}' for i, f in enumerate(functions)])
        ax.set_title(plot_title)
        ax.legend()
        fig.tight_layout()

        for widget in graph_canvas.winfo_children():
            widget.destroy()

        canvas = FigureCanvas(fig, master=graph_canvas)
        canvas.draw()
        canvas.get_tk_widget().pack(side="top", fill="both", expand=True)

        if toolbar:
            toolbar.destroy()

        toolbar = NavigationToolbar2Tk(canvas, graph_canvas)
        toolbar.update()
        toolbar.pack(side="bottom", fill="x")

    except Exception as e:
        messagebox.showerror("Graph Error", f"Could not plot function(s):\n{e}")

def setup_empty_graph():
    fig = Figure(figsize=(6, 4), dpi=100)  # Match plot_function dimensions
    ax = fig.add_subplot(111)

    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    ax.axhline(0, color='black', linewidth=1)  # Draw x-axis
    ax.axvline(0, color='black', linewidth=1)  # Draw y-axis
    ax.grid(True, which='both')

    global graph_canvas  # Assuming you have a frame/container
    for widget in graph_canvas.winfo_children():
        widget.destroy()

    canvas = FigureCanvas(fig, master=graph_canvas)  # Important: use FigureCanvasTkAgg
    canvas.draw()
    canvas.get_tk_widget().pack(side="top", fill="both", expand=True)

    toolbar = NavigationToolbar2Tk(canvas, graph_canvas)
    toolbar.update()
    toolbar.pack(side="bottom", fill="x")



'''
Standard Evaluate function
'''

def evaluate_input():
    input_str = text_entry.get()
    text = preprocess_input(input_str)
    try:
        expr = sp.sympify(convert_to_sympy_value(text), locals=KNOWN_CONSTANTS)

        if second is not True:
            output = sp.simplify(expr)
            if output.is_Integer:
                result = output
            elif output.is_Rational:
                result = output
            else:
                result = output.evalf()
        else:
            result = expr.evalf()

        global output_label
        output_label.destroy()
        latex_output = render_latex(sp.latex(result).replace(r'\log', r'\ln'))
        output_label = ctk.CTkLabel(master=root, image=latex_output, text='', height=100, width=100,
                                    fg_color=main_color)
        output_label.place(x=155, y=15)

    except Exception as e:
        messagebox.showerror("Invalid Input", f"Input could not be understood.\n\n{e}")


'''
Calculus Functions
'''

def find_derivative():
    string = text_entry.get()
    text = preprocess_input(string)
    try:
        if calc_var_input.get() != '':
            symbol = sp.Symbol(calc_var_input.get())
        else:
            symbol = sp.Symbol('x')
        expr = convert_to_sympy_value(text)
        output = sp.diff(expr, symbol)
        result = sp.latex(output)
        image = render_latex(result)
        output_label.configure(image=image, text='')
        output_label.image = image
    except ValueError:
        messagebox.showerror("Invalid Input", "Input could not be understood, please try again")

def find_antiderivative():
    string = text_entry.get()
    text = preprocess_input(string)
    try:
        if second is False:
            if calc_var_input.get() != '':
                symbol = sp.Symbol(calc_var_input.get())
            else:
                symbol = sp.Symbol('x')
            expr = convert_to_sympy_value(text)
            output = sp.integrate(expr, symbol)
            result = sp.latex(output).replace(r'\log', r'\ln')
            image = render_latex(result)
            output_label.configure(image=image, text='')
            output_label.image = image
        elif second is True:
            a = extra_input1.get()
            b = extra_input2.get()
            if calc_var_input.get() != '':
                symbol = sp.Symbol(calc_var_input.get())
            else:
                symbol = sp.Symbol('x')
            expr = convert_to_sympy_value(text)
            output = sp.integrate(expr, (symbol, a, b))
            result = sp.latex(output).replace(r'\log', r'\ln')
            image = render_latex(result)
            output_label.configure(image=image, text='')
            output_label.image = image
    except ValueError:
        messagebox.showerror("Invalid Input", "Input could not be understood, please try again")

'''
Linear Algebra functions
'''
def linalg_menu():
    try:
        popup = ctk.CTkToplevel()
        popup.geometry("200x125")
        popup.configure(fg_color=main_color)
        popup.title("Matrix")
        popup.transient(root)
        popup.focus_set()
        apply_style(popup, "dark")

        create_matrix_button = ctk.CTkButton(master=popup, text="Create Matrix", command=create_matrix,
                                             text_color=button_text_color,
                                             fg_color=button_color)
        create_matrix_button.pack(padx=20, pady=5)

        clear_matrix_button = ctk.CTkButton(master=popup, text="Clear Matrix Mem", command=clear_matrix,
                                            text_color=button_text_color,
                                            fg_color=button_color)
        clear_matrix_button.pack(padx=20, pady=5)

        multiply_matrix_button = ctk.CTkButton(master=popup, text="[ : : ][ : : ]", command=multiply_matrix,
                                               text_color=button_text_color,
                                               fg_color=button_color)
        multiply_matrix_button.pack(padx=20, pady=5)

    except Exception as e:
        messagebox.showerror("Memory Error", f'Error occurred: {e}')

# Add matrix to memory location
def create_matrix():
    try:
        global matrix_mem, matrix_mem2
        text = text_entry.get()
        rows = int(extra_input1.get())
        columns = int(extra_input2.get())
        val_list = np.array([float(v.strip()) for v in text.split(',') if v.strip() != ''])
        matrix = val_list.reshape(rows, columns)
        if matrix_mem is None:
            matrix_mem = matrix
            matrix_mem_label.configure(text=f"{matrix}")
        elif matrix_mem2 is None:
            matrix_mem2 = matrix
            matrix_mem_label2.configure(text=f"{matrix}")
    except ValueError:
        messagebox.showerror("Invalid Input", "Input could not be understood, please try again")

# Multiply currently stored matrices
def multiply_matrix():
    try:
        global matrix_mem, matrix_mem2
        if matrix_mem is not None and matrix_mem2 is not None:
            result = np.matmul(matrix_mem, matrix_mem2)
            # Remove image by re-creating the label entirely
            global output_label
            output_label.destroy()
            output_label = ctk.CTkLabel(master=root, text=f'{result}', font=("Roboto", 15), height=100, width=100, fg_color=main_color,
                                        text_color=background_text_color)
            output_label.place(x=155, y=15)
    except ValueError:
        messagebox.showerror('Invalid Input', 'Input could not be understood, check matrix dimensions')

'''
Trig Functions
'''

# Sin(x)
def insert_sin():
    try:
        if second is False:
            text_entry.insert("end", "sin(")
        elif second is True:
            text_entry.insert("end", "asin(")
    except ValueError:
        messagebox.showerror("Invalid Input", "Input could not be understood, please try again")

# Cos(x)
def insert_cos():
    try:
        if second is False:
            text_entry.insert("end", "cos(")
        elif second is True:
            text_entry.insert("end", "acos(")
    except ValueError:
        messagebox.showerror("Invalid Input", "Input could not be understood, please try again")

# Tan(x)
def insert_tan():
    try:
        if second is False:
            text_entry.insert("end", "tan(")
        elif second is True:
            text_entry.insert("end", "atan(")
    except ValueError:
        messagebox.showerror("Invalid Input", "Input could not be understood, please try again")

'''
Probability Functions
'''
#combinations
def nCr():
    if second is False:
        text_entry.insert("end", "nCr(")
    elif second is True:
        text_entry.insert("end", "nPr(")

"""
Algebra Functions
"""
def alg_menu():
    try:
        popup = ctk.CTkToplevel()
        popup.geometry("200x125")
        popup.configure(fg_color=main_color)
        popup.title("Algebra")
        popup.transient(root)
        popup.focus_set()
        apply_style(popup, "dark")

        solve_button = ctk.CTkButton(master=popup, text="Solve", command=alg_solve,
                                             text_color=button_text_color,
                                             fg_color=button_color)
        solve_button.pack(padx=20, pady=5)

        factor_button = ctk.CTkButton(master=popup, text="Factor", command=alg_factor,
                                            text_color=button_text_color,
                                            fg_color=button_color)
        factor_button.pack(padx=20, pady=5)

        inequality_button = ctk.CTkButton(master=popup, text='Inequality', command=alg_inequality,
                                               text_color=button_text_color,
                                               fg_color=button_color)
        inequality_button.pack(padx=20, pady=5)

    except Exception as e:
        messagebox.showerror("Memory Error", f'Error occurred: {e}')


def alg_solve():
    input_text = text_entry.get().strip()  # Get the input text and remove any surrounding whitespace

    # Preprocess the input (e.g., handle nCr, nPr, etc.)
    text = preprocess_input(input_text)

    # Replace any single equal signs with double equal signs for SymPy compatibility
    input_text = input_text.replace("=", "==")

    try:
        # Check if the user has specified a variable; if not, default to 'x'
        if calc_var_input.get().strip() != '':
            symbol = sp.Symbol(calc_var_input.get().strip())
        else:
            symbol = sp.Symbol('x')

        # Handle input expressions and ensure proper formatting before SymPy processing
        # Split the input into left and right sides if it's an equation
        if '==' in input_text:
            left, right = input_text.split('==')

            # Convert both left and right sides to SymPy expressions
            try:
                left_expr = sp.sympify(convert_to_sympy_value(left.strip()), locals=KNOWN_CONSTANTS)
                print("Left Expression after sympify:", left_expr)
            except Exception as e:
                messagebox.showerror("Input Error", f"Error Occurred: {e}")

            try:
                right_expr = sp.sympify(convert_to_sympy_value(right.strip()), locals=KNOWN_CONSTANTS)
                print("Right Expression after sympify:", right_expr)
            except Exception as e:
                messagebox.showerror("Input Error", f"Error Occurred: {e}")

            # Create the equation
            equation = sp.Eq(left_expr, right_expr)
        else:
            # If there's no equality sign, solve the equation as expr = 0
            expr = sp.sympify(convert_to_sympy_value(text), locals=KNOWN_CONSTANTS)
            equation = sp.Eq(expr, 0)

        # Solve the equation
        output = sp.solve(equation, symbol)

        # Display the result
        global output_label
        output_label.destroy()
        latex_output = render_latex(sp.latex(output).replace(r'\log', r'\ln'))  # Render the LaTeX output
        output_label = ctk.CTkLabel(master=root, image=latex_output, text='', height=100, width=100,
                                    fg_color=main_color)
        output_label.place(x=155, y=15)

    except Exception as e:
        # Display an error message if anything goes wrong
        messagebox.showerror("Input Error", f"Error Occurred: {e}")


from sympy import solveset, S

def alg_inequality():
    input_text = text_entry.get().strip()
    text = preprocess_input(input_text)

    input_text = input_text.replace("=", "==")

    try:
        if calc_var_input.get().strip() != '':
            symbol = sp.Symbol(calc_var_input.get().strip())
        else:
            symbol = sp.Symbol('x')

        # Detect if it's inequality or equation
        if any(op in input_text for op in ['>', '<', '>=', '<=']):
            expr = sp.sympify(convert_to_sympy_value(text), locals=KNOWN_CONSTANTS)
            output = sp.solveset(expr, symbol, domain=S.Reals)
        else:
            if '==' in input_text:
                left, right = input_text.split('==')
                left_expr = sp.sympify(convert_to_sympy_value(left.strip()), locals=KNOWN_CONSTANTS)
                right_expr = sp.sympify(convert_to_sympy_value(right.strip()), locals=KNOWN_CONSTANTS)
                equation = sp.Eq(left_expr, right_expr)
            else:
                equation = sp.sympify(convert_to_sympy_value(text), locals=KNOWN_CONSTANTS)
            output = sp.solve(equation, symbol)

        global output_label
        output_label.destroy()
        latex_output = render_latex(sp.latex(output).replace(r'\log', r'\ln'))
        output_label = ctk.CTkLabel(master=root, image=latex_output, text='', height=100, width=100,
                                    fg_color=main_color)
        output_label.place(x=155, y=15)

    except Exception as e:
        messagebox.showerror("Input Error", f"Error Occurred: {e}")

def alg_factor():
    input_text = text_entry.get().strip()
    text = preprocess_input(input_text)

    try:
        expr = sp.sympify(convert_to_sympy_value(text), locals=KNOWN_CONSTANTS)
        output = sp.factor(expr)

        global output_label
        output_label.destroy()
        latex_output = render_latex(sp.latex(output).replace(r'\log', r'\ln'))
        output_label = ctk.CTkLabel(master=root, image=latex_output, text='', height=100, width=100,
                                    fg_color=main_color)
        output_label.place(x=155, y=15)

    except Exception as e:
        messagebox.showerror("Input Error", f"Error Occurred: {e}")



"""
Memory Functions
"""
# Clear all matrices from memory locations
def clear_matrix():
    global matrix_mem, matrix_mem2
    matrix_mem, matrix_mem2 = None, None
    matrix_mem_label.configure(text="")
    matrix_mem_label2.configure(text="")

def mem_store():
    global mem1, mem2, mem3
    input_str = text_entry.get()
    text = preprocess_input(input_str)
    try:
        expr = sp.sympify(convert_to_sympy_value(text), locals=KNOWN_CONSTANTS)

        if second is not True:
            output = sp.simplify(expr)
            if output.is_Integer:
                result = output
            elif output.is_Rational:
                result = output
            else:
                result = output.evalf()
        else:
            result = expr.evalf()
        if mem1 is None:
            mem1 = result
        elif mem2 is None:
            mem2 = result
        elif mem3 is None:
            mem3 = result
        else:
            mem1 = result
    except Exception as e:
        messagebox.showerror("Memory Error", f"Error occurred: {e}")

def mem_clear():
    try:
        global mem1, mem2, mem3
        mem1, mem2, mem3 = None, None, None
    except Exception as e:
        messagebox.showerror("Memory Error", f"Error occurred: {e}")

def mem_recall():
    global mem1, mem2, mem3, popup
    try:
        popup = ctk.CTkToplevel()
        popup.geometry("200x125")
        popup.configure(fg_color=main_color)
        popup.title("Memory Recall")
        popup.transient(root)
        popup.focus_set()
        apply_style(popup, "dark")

        mem1_button = ctk.CTkButton(master=popup, command=lambda: text_entry.insert("end", mem1),
                                    text=f'Mem1 = {mem1}', text_color="black", fg_color=button_color)
        mem1_button.pack(padx=20, pady=5)

        mem2_button = ctk.CTkButton(master=popup, command=lambda: insert_char(mem2),
                                    text=f'Mem2 = {mem2}', text_color="black", fg_color=button_color)
        mem2_button.pack(padx=20, pady=5)

        mem3_button = ctk.CTkButton(master=popup, command=lambda: text_entry.insert("end", mem3),
                                    text=f'Mem2 = {mem3}', text_color="black", fg_color=button_color)
        mem3_button.pack(padx=20, pady=5)

    except Exception as e:
        messagebox.showerror("Memory Error", f'Error occurred: {e}')


'''''''''''''''''''''''''''''''''''''''

Initialize main root window and widgets

'''''''''''''''''''''''''''''''''''''''
root = ctk.CTk()
root.title("Graphing Calculator")
root.geometry("750x500")
root.configure(fg_color=main_color)
apply_style(root, style="dark")


'''
Frames
'''
keypad_frame = ctk.CTkFrame(master=root, fg_color=secondary_color)
keypad_frame.place(x=20, y=150)

'''
Text Entries
'''
text_entry = ctk.CTkEntry(master=root, fg_color=secondary_color, text_color=background_text_color,
                          border_color=secondary_color)
text_entry.place(x=10, y=50)
text_entry.bind("<FocusIn>", lambda e: set_focus(text_entry))

extra_input1 = ctk.CTkEntry(master=root, height=30, width=30, fg_color=secondary_color, border_color=secondary_color,
                            text_color=background_text_color)
extra_input1.place(x=10, y=20)
extra_input1.bind("<FocusIn>", lambda e: set_focus(extra_input1))

extra_input2 = ctk.CTkEntry(master=root, height=30, width=30, fg_color=secondary_color, border_color=secondary_color,
                            text_color=background_text_color)
extra_input2.place(x=40, y=20)
extra_input2.bind("<FocusIn>", lambda e: set_focus(extra_input2))

calc_var_input = ctk.CTkEntry(master=root, height=30, width=30, fg_color=secondary_color, border_color=secondary_color,
                              text_color=background_text_color)
calc_var_input.place(x=120, y=20)
calc_var_input.bind("<FocusIn>", lambda e: set_focus(calc_var_input))
'''
Output labels
'''
output_label = ctk.CTkLabel(master=root, text="", height = 100, width=100, fg_color=main_color)
output_label.place(x=185, y=15)

graph_canvas = ctk.CTkCanvas(root, width=600, height=400, highlightthickness=0, bg=main_color)
graph_canvas.place(x=325, y=150)

'''
Other labels
'''
matrix_mem_label = ctk.CTkLabel(master=root, text='', fg_color=main_color, text_color=background_text_color)
matrix_mem_label.place(x=10, y=450)

matrix_mem_label2 = ctk.CTkLabel(master=root, text='', fg_color=main_color, text_color=background_text_color)
matrix_mem_label2.place(x=110, y=450)

calc_label = ctk.CTkLabel(master=root, text="Variable →", font=("Roboto", 9), text_color=background_text_color,
                          anchor='e', justify="right", fg_color=main_color)
calc_label.place(x=75, y=21)

parameter_label = ctk.CTkLabel(master=root, text="Parameter Input", font=("Roboto", 8),
                               text_color=background_text_color, height=10)
parameter_label.place(x=13, y=5)

'''
Buttons for functions and evaluation
'''

'''
Second button
'''
second_button = ctk.CTkButton(master = root, text='2nd', command=toggle_second, text_color=button_text_color,
                              width=100, height=30, fg_color=button_color)
second_button.place(x=140, y=400)

'''
Evaluation button
'''
eval_button = ctk.CTkButton(master= root, command=evaluate_input, text="Enter (=)", width=110,
                            text_color=button_text_color, fg_color=button_color)
eval_button.place(x=24, y=80)

'''
Graphing button
'''
plot_button = ctk.CTkButton(master=root, text="Graph", command=plot_function, text_color=button_text_color,
                            fg_color=button_color, width=110)
plot_button.place(x=24, y=117)


'''
Calculus buttons
'''
derivative_button = ctk.CTkButton(master=root, command=find_derivative, text="d/dx", height=30,
                                  width=30, text_color=button_text_color, fg_color=button_color)
derivative_button.place(x=60, y=400)

antiderivative_button = ctk.CTkButton(master=root, command=find_antiderivative, text="∫dx", height=30,
                                  width=30, text_color=button_text_color, fg_color=button_color)
antiderivative_button.place(x=25, y=400)

'''
Linear Algebra buttons
'''
linalg_menu_button = ctk.CTkButton(master=root, text="[ : : ]", command=linalg_menu, text_color=button_text_color,
                                   height=30, width=30, fg_color=button_color)
linalg_menu_button.place(x=145, y=155)

'''
Trig buttons
'''
sine_button = ctk.CTkButton(master=root, text='sin(x)', command=insert_sin, text_color=button_text_color,
                            height=30, width=30,
                            fg_color=button_color)
sine_button.place(x=145, y=275)

cosine_button = ctk.CTkButton(master=root, text='cos(x)', command=insert_cos, text_color=button_text_color,
                              height=30, width=30,
                              fg_color=button_color)
cosine_button.place(x=195, y=275)

tangent_button = ctk.CTkButton(master=root, text='tan(x)', command=insert_tan, text_color=button_text_color,
                               height=30, width=30,
                               fg_color=button_color)
tangent_button.place(x=145, y=235)

'''
Algebraic function buttons
'''
pi_button = ctk.CTkButton(master=root, text='π', command=insert_pi, text_color=button_text_color,
                          height=30, width=30,
                          fg_color=button_color)
pi_button.place(x=145, y=315)

equal_button = ctk.CTkButton(master=root, text="=", command=lambda e: insert_char("="), text_color=button_text_color,
                             fg_color=button_color, height=30, width=30)
equal_button.place(x=205, y=195)

e_button = ctk.CTkButton(master=root, text='eˣ', command=insert_e, text_color=button_text_color,
                         height=30, width=30,
                         fg_color=button_color)
e_button.place(x=177, y=315)

infinity_button = ctk.CTkButton(master=root, text='∞', command=insert_infinity, text_color=button_text_color,
                                height=30, width=30,
                                fg_color=button_color)
infinity_button.place(x=209, y=315)

radical_button = ctk.CTkButton(master=root, text='√', command=insert_radical, text_color=button_text_color,
                               height=30, width=30,
                               fg_color=button_color)
radical_button.place(x=177, y=355)

i_button = ctk.CTkButton(master=root, text='i', command=insert_i, text_color=button_text_color, height=30, width=30,
                         fg_color=button_color)
i_button.place(x=209, y=355)

alg_menu_button = ctk.CTkButton(master=root, text="Algebra", command=alg_menu, text_color=button_text_color,
                                width=55, height=31, fg_color=button_color)
alg_menu_button.place(x=145, y=195)

'''
Probability Buttons
'''
nCr_button = ctk.CTkButton(master=root, text='nCr', command=nCr, text_color=button_text_color, height=30, width=50,
                       fg_color=button_color)
nCr_button.place(x=195, y=235)
'''
Number and character keypad
'''
buttons = [
    ['7', '8', '9'],
    ['4', '5', '6'],
    ['1', '2', '3'],
    ['+', '0', '-'],
    ['*', '/', '^'],
    ['(', ')', 'x'],
]

for r, row in enumerate(buttons, start=1):
    for c, char in enumerate(row):
        btn = ctk.CTkButton(master=keypad_frame, text=char, command=lambda ch=char: insert_char(ch),
                            width=30, height=30, text_color=button_text_color, fg_color=button_color)
        btn.grid(row=r, column=c, padx=5, pady=5)

# Backspace button
del_button = ctk.CTkButton(master=root, command=backspace, text="<-", text_color=button_text_color,
                           width=30, height=30, fg_color=button_color)
del_button.place(x=105, y=400)

# Clear all button
clear_button = ctk.CTkButton(master=root, command=clear, text='CE', text_color=button_text_color,
                             height = 30, width=30, fg_color=button_color)
clear_button.place(x=145, y=355)

"""
Memory Buttons
"""
# Mem Store button
store_button = ctk.CTkButton(master=root, command=mem_store, text='Mem', text_color=button_text_color,
                             height = 28, width=30, fg_color=button_color)
store_button.place(x=135, y=117)

# Mem Clear button
store_button = ctk.CTkButton(master=root, command=mem_clear, text='M.CE', text_color=button_text_color,
                             height = 28, width=30, fg_color=button_color)
store_button.place(x=185, y=155)

# Mem Recall button
recall_button = ctk.CTkButton(master=root, command=mem_recall, text='M.Re', text_color=button_text_color,
                              height = 28, width=30, fg_color=button_color)
recall_button.place(x=180, y=117)
''' 
Window protocol and startup
'''
#Protocol and window startup
root.protocol("WM_DELETE_WINDOW", on_closing)
setup_empty_graph()
root.mainloop()