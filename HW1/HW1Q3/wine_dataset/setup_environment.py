"""
Environment setup script for Wine Quality Analysis
Run this first to install required packages
"""

import subprocess
import sys
import os

def install_packages():
    """Install required packages from requirements.txt"""
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    requirements_path = os.path.join(script_dir, "requirements.txt")
    
    print(f"Looking for requirements.txt at: {requirements_path}")
    
    if not os.path.exists(requirements_path):
        print("✗ requirements.txt not found! Creating a basic one...")
        # Create a basic requirements file
        with open(requirements_path, 'w') as f:
            f.write("numpy\npandas\nscipy\nmatplotlib\nscikit-learn\n")
        print("✓ Created requirements.txt")
    
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", requirements_path])
        print("✓ All packages installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Error installing packages: {e}")
        return False

def verify_environment():
    """Verify that all required packages are available"""
    packages = ['numpy', 'pandas', 'scipy', 'matplotlib', 'sklearn']
    
    print("Verifying environment...")
    for package in packages:
        try:
            if package == 'sklearn':
                __import__('sklearn')
            else:
                __import__(package)
            print(f"✓ {package} is available")
        except ImportError:
            print(f"✗ {package} is NOT available")
            return False
    
    print("\n🎉 Environment is ready!")
    print("You can now run: python wine_analysis.py")
    return True

if __name__ == "__main__":
    print("Setting up environment for Wine Quality Analysis...")
    print(f"Python version: {sys.version}")
    print("Installing packages from requirements.txt...")
    
    if install_packages():
        verify_environment()
    else:
        print("Please check your internet connection and try again.")