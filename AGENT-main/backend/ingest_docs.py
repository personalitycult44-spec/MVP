import os
import glob
import requests

# =====================================================================
# PRODUCTION ENDPOINT TARGET CONFIGURATION
# =====================================================================
TARGET_URL = "https://agent-production-0172.up.railway.app/api/v1/admin/upload-pdf"
PDF_FOLDER_PATH = "./backend/pdfs"  # Matches your local directory path

def run_knowledge_ingestion():
    search_path = os.path.join(PDF_FOLDER_PATH, "*.pdf")
    pdf_files = glob.glob(search_path)
    
    if not pdf_files:
        print(f"❌ No PDF documents found matching search pattern at: {PDF_FOLDER_PATH}")
        print("Please ensure your PDF files are placed inside your backend/pdfs/ folder.")
        return

    print(f"🚀 Found {len(pdf_files)} documentation assets target paths. Starting ingestion pipeline...\n")

    # Adding common browser headers to bypass strict local web shield firewalls
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Connection": "keep-alive"
    }

    for file_path in pdf_files:
        file_name = os.path.basename(file_path)
        print(f"📦 Processing: {file_name}")
        
        try:
            with open(file_path, "rb") as pdf_file:
                files = {
                    "file": (file_name, pdf_file, "application/pdf")
                }
                
                # OPTIMIZATION: Added stream=True to prevent packet hoarding on local hosts
                response = requests.post(
                    TARGET_URL, 
                    files=files, 
                    headers=headers,
                    stream=True, 
                    timeout=300
                )
                
                if response.status_code in [200, 201]:
                    print(f"✅ Successfully ingested: {file_name}")
                    print(f"   Server Response: {response.json()}\n")
                else:
                    print(f"❌ Failed to ingest: {file_name} (Status Code: {response.status_code})")
                    print(f"   Error Payload: {response.text}\n")
                    
        except requests.exceptions.ConnectionError as conn_err:
            print(f"🚨 Connection Interrupted for {file_name}: Your host machine software aborted the stream.")
            print("👉 Fix: Ensure any active third-party web shield, antivirus, or VPN proxy is briefly paused.\n")
        except requests.exceptions.Timeout:
            print(f"🚨 Network Timeout: Processing {file_name} took longer than 5 minutes.\n")
        except Exception as e:
            print(f"🚨 Unexpected failure while uploading {file_name}: {str(e)}\n")

    print("🏁 Knowledge base ingestion cycle sequence finished.")

if __name__ == "__main__":
    run_knowledge_ingestion()