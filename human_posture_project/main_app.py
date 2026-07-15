import subprocess
import sys

print("\n=== Human Posture Analysis System ===")
print("1. Gym Exercise Analysis")
print("2. Sitting Posture Analysis")

choice = input("Enter choice (1 or 2): ").strip()

if choice == "1":
    subprocess.run([sys.executable, "final_gym_app.py"])
elif choice == "2":
    subprocess.run([sys.executable, "final_posture_app.py"])
else:
    print("Invalid choice")
