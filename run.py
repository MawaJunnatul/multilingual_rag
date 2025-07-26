import os
import sys
import argparse
import subprocess

def install_dependencies():
    """Install required Python packages."""
    print("📦 Installing dependencies...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    print("✅ Dependencies installed successfully!")

def setup_environment():
    """Set up environment variables from .env file."""
    if not os.path.exists('.env'):
        print("⚠️  .env file not found. Creating a default one...")
        with open('.env', 'w') as f:
            f.write("COHERE_API_KEY=your_cohere_api_key_here\n")
        print("📝 Please update .env with your actual Cohere API key.")
        return False
    return True

def run_streamlit():
    """Run the Streamlit user interface."""
    print("🚀 Launching Streamlit app...")
    subprocess.run([sys.executable, "-m", "streamlit", "run", "ui/streamlit_app.py"])

def run_api():
    """Run the FastAPI server."""
    print("🚀 Starting FastAPI server...")
    subprocess.run([sys.executable, "api/app.py"])

def run_evaluation(auto_build=False):
    """Run the evaluation script."""
    print("🔍 Running evaluation suite...")
    cmd = [sys.executable, "-m", "src.evaluation"]
    if auto_build:
        cmd.append("--auto-build")
    subprocess.run(cmd)

def print_help():
    print("🧠 Multilingual RAG System")
    print("Usage:")
    print("  python run.py --install         Install dependencies")
    print("  python run.py --setup           Setup environment (.env)")
    print("  python run.py --ui              Run Streamlit UI")
    print("  python run.py --api             Run FastAPI server")
    print("  python run.py --evaluation      Run evaluation suite")
    print("       [--auto-build]            Auto-build knowledge base if missing")

def main():
    parser = argparse.ArgumentParser(description="Multilingual RAG System", add_help=False)
    parser.add_argument("--install", action="store_true", help="Install dependencies")
    parser.add_argument("--setup", action="store_true", help="Setup environment")
    parser.add_argument("--ui", action="store_true", help="Run Streamlit UI")
    parser.add_argument("--api", action="store_true", help="Run FastAPI server")
    parser.add_argument("--evaluation", action="store_true", help="Run evaluation suite")
    parser.add_argument("--auto-build", action="store_true", help="Auto-build knowledge base (used with --evaluation)")
    
    args = parser.parse_args()

    if args.install:
        install_dependencies()
    elif args.setup:
        setup_environment()
    elif args.ui:
        if setup_environment():
            run_streamlit()
    elif args.api:
        if setup_environment():
            run_api()
    elif args.evaluation:
        run_evaluation(auto_build=args.auto_build)
    else:
        print_help()

if __name__ == "__main__":
    main()
