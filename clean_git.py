import shutil, os, subprocess, time

remote_url = "https://github.com/dhruvsolanki30/AutoSAM-X"

# Get remote URL
r = subprocess.run(['git', 'remote', 'get-url', 'origin'], capture_output=True, text=True)
if r.returncode == 0:
    remote_url = r.stdout.strip()

print(f"Remote: {remote_url}")

# Close any git processes
subprocess.run(["taskkill", "/F", "/IM", "git.exe"], capture_output=True)
time.sleep(2)

# Remove .git with retry
print("Removing .git...")
for attempt in range(3):
    try:
        if os.path.exists('.git'):
            shutil.rmtree('.git', ignore_errors=True)
        break
    except Exception as e:
        print(f"Attempt {attempt+1} failed: {e}")
        time.sleep(2)

print("✓ Creating fresh git repo...")
subprocess.run(['git', 'init'], check=True)
subprocess.run(['git', 'config', 'user.name', 'AutoSAM-X'], check=True)
subprocess.run(['git', 'config', 'user.email', 'user@example.com'], check=True)
subprocess.run(['git', 'add', '.'], check=True)
subprocess.run(['git', 'commit', '-m', 'Clean multi-pathology framework'], check=True)
subprocess.run(['git', 'remote', 'add', 'origin', remote_url], check=True)
subprocess.run(['git', 'branch', '-M', 'feature-lungs'], check=True)
print("✓ Fresh clean repository ready")
