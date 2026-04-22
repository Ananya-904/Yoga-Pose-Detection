"""
Tkinter Sign-In / Sign-Up system with JSON persistence.
Code below implements a simple authentication flow using Tkinter for the UI and JSON for storing user data. It includes:
- Sign-In Window: Allows existing users to log in with email and password.
"""

import json
import os
import re
import tkinter as tk
from tkinter import ttk, messagebox

from app import YogaAIApp

USERS_FILE = "users.json"
EMAIL_PATTERN = re.compile(r"^[\w\.-]+@[\w\.-]+\.\w+$")


def ensure_users_file():
    """Ensure the JSON file exists."""
    if not os.path.exists(USERS_FILE):
        with open(USERS_FILE, "w", encoding="utf-8") as f:
            json.dump([], f, indent=2)


def load_users():
    """Load all registered users."""
    ensure_users_file()
    with open(USERS_FILE, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
            return data if isinstance(data, list) else []
        except json.JSONDecodeError:
            return []


def save_users(users):
    """Writes the updated list back to JSON.
    Persist users to JSON."""
    with open(USERS_FILE, "w", encoding="utf-8") as f:
        json.dump(users, f, indent=2)


def email_exists(email):
    """Check if email already registered."""
    email = email.lower()
    return any(user["email"] == email for user in load_users())


def validate_email(email):
    """Uses REGEX to check if email is valid:
    Validate email format."""
    return bool(EMAIL_PATTERN.match(email))


class SignInWindow:
    """Login window."""

    def __init__(self, root):
        self.root = root
        self.root.title("Yoga AI | Sign In")
        self.root.geometry("420x380")
        self.root.resizable(False, False)
        self.root.configure(bg="#1e272e")

        self.email_var = tk.StringVar()
        self.password_var = tk.StringVar()

        self.build_ui()

    def build_ui(self):
        title = tk.Label(
            self.root,
            text="Welcome Back",
            font=("Arial", 20, "bold"),
            bg="#1e272e",
            fg="white",
        )
        title.pack(pady=20)

        card = tk.Frame(self.root, bg="#2f3640", padx=25, pady=25, bd=0)
        card.pack(padx=20, pady=5, fill=tk.BOTH, expand=True)

        tk.Label(card, text="Email", bg="#2f3640", fg="#dcdde1").pack(anchor="w")
        email_entry = ttk.Entry(card, textvariable=self.email_var)
        email_entry.pack(fill=tk.X, pady=(0, 10))

        tk.Label(card, text="Password", bg="#2f3640", fg="#dcdde1").pack(anchor="w")
        password_entry = ttk.Entry(card, textvariable=self.password_var, show="*")
        password_entry.pack(fill=tk.X, pady=(0, 15))

        signin_btn = ttk.Button(card, text="Sign In", command=self.handle_sign_in)
        signin_btn.pack(fill=tk.X, pady=5)

        divider = tk.Frame(card, height=1, bg="#353b48")
        divider.pack(fill=tk.X, pady=10)

        signup_prompt = tk.Label(
            card,
            text="Don't have an account?",
            bg="#2f3640",
            fg="#dcdde1",
        )
        signup_prompt.pack()
        signup_btn = ttk.Button(card, text="Create Account", command=self.open_sign_up)
        signup_btn.pack(fill=tk.X, pady=5)

    def open_sign_up(self):
        sign_up_window(self.root)

    def handle_sign_in(self):
        email = self.email_var.get().strip().lower()
        password = self.password_var.get().strip()

        if not email or not password:
            messagebox.showwarning("Missing Fields", "Please fill out all fields.")
            return

        if not validate_email(email):
            messagebox.showerror("Invalid Email", "Please enter a valid email address.")
            return

        users = load_users()
        user = next((u for u in users if u["email"] == email), None)

        if not user or user["password"] != password:
            messagebox.showerror("Login Failed", "Invalid email or password.")
            return

        messagebox.showinfo("Success", f"Welcome, {user['name']}!")
        self.email_var.set("")
        self.password_var.set("")

        # Hide sign-in window and open Yoga AI home UI
        self.root.withdraw()
        home_window(self.root, user)


class SignUpWindow:
    """Registration window."""

    def __init__(self, parent):
        self.parent = parent
        self.window = tk.Toplevel(parent)
        self.window.title("Create Account")
        self.window.geometry("460x480")
        self.window.resizable(False, False)
        self.window.configure(bg="#1e272e")

        self.name_var = tk.StringVar()
        self.email_var = tk.StringVar()
        self.password_var = tk.StringVar()
        self.confirm_var = tk.StringVar()

        self.build_ui()

    def build_ui(self):
        title = tk.Label(
            self.window,
            text="Create Your Account",
            font=("Arial", 20, "bold"),
            bg="#1e272e",
            fg="white",
        )
        title.pack(pady=20)

        card = tk.Frame(self.window, bg="#2f3640", padx=25, pady=25)
        card.pack(padx=20, pady=5, fill=tk.BOTH, expand=True)

        for label_text, var, show in [
            ("Name", self.name_var, None),
            ("Email", self.email_var, None),
            ("Password", self.password_var, "*"),
            ("Confirm Password", self.confirm_var, "*"),
        ]:
            tk.Label(card, text=label_text, bg="#2f3640", fg="#dcdde1").pack(
                anchor="w"
            )
            entry = ttk.Entry(card, textvariable=var, show=show)
            entry.pack(fill=tk.X, pady=(0, 10))

        register_btn = ttk.Button(card, text="Sign Up", command=self.handle_sign_up)
        register_btn.pack(fill=tk.X, pady=5)

        already_btn = ttk.Button(
            card, text="Back to Sign In", command=self.close_and_focus_sign_in
        )
        already_btn.pack(fill=tk.X, pady=5)

        self.window.grab_set()
        self.window.focus_force()
        self.window.protocol("WM_DELETE_WINDOW", self.close_and_focus_sign_in)

    def close_and_focus_sign_in(self):
        self.window.destroy()
        self.parent.deiconify()
        self.parent.focus_force()

    def handle_sign_up(self):
        name = self.name_var.get().strip()
        email = self.email_var.get().strip().lower()
        password = self.password_var.get().strip()
        confirm = self.confirm_var.get().strip()

        if not all([name, email, password, confirm]):
            messagebox.showwarning("Missing Fields", "Please fill out every field.")
            return

        if not validate_email(email):
            messagebox.showerror("Invalid Email", "Please enter a valid email address.")
            return

        if password != confirm:
            messagebox.showerror("Password Mismatch", "Passwords do not match.")
            return

        if email_exists(email):
            messagebox.showerror("Duplicate Email", "This email is already registered.")
            return

        users = load_users()
        users.append({"name": name, "email": email, "password": password})
        save_users(users)

        messagebox.showinfo("Success", "Account created successfully! Please sign in.")
        self.close_and_focus_sign_in()


def sign_up_window(parent):
    """Open the sign-up UI."""
    SignUpWindow(parent)


def home_window(parent, user):
    """Open the Yoga AI main application after login."""
    yoga_window = tk.Toplevel(parent)

    def on_logout():
        messagebox.showinfo("Logged Out", "You have been logged out.")
        yoga_window.destroy()
        parent.deiconify()
        parent.focus_force()

    YogaAIApp(yoga_window, on_logout=on_logout)


def sign_in_window():
    """Entry point for the auth system."""
    root = tk.Tk()
    SignInWindow(root)
    root.mainloop()


if __name__ == "__main__":
    sign_in_window()



