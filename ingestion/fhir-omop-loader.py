import os
import glob
import re
import json
import random
import requests
from datetime import datetime

# FHIR Server configuration (cluster-internal DNS)
FHIR_URL = os.environ.get("FHIR_URL", "http://colabbio-hapi-fhir.colabbio-dev.svc.cluster.local:8080/fhir")
WSI_MOUNT_DIR = os.environ.get("WSI_MOUNT_DIR", "/mnt/wsi_data/BRCA_AWS/BRCA_raw")

# Headers for FHIR API
HEADERS = {
    'Content-Type': 'application/fhir+json'
}

def generate_synthetic_patient(patient_id):
    """Generate a synthetic FHIR Patient resource."""
    # Synthetic demographics
    gender = random.choice(["female", "female", "female", "male"]) # BRCA predominantly female
    age = random.randint(35, 80)
    birth_year = datetime.now().year - age
    
    return {
        "resourceType": "Patient",
        "id": patient_id,
        "identifier": [
            {
                "system": "http://gdc.cancer.gov/tcga",
                "value": patient_id
            }
        ],
        "gender": gender,
        "birthDate": f"{birth_year}-01-01",
        "name": [
            {
                "use": "usual",
                "family": f"Synthetic_{patient_id}",
                "given": ["Patient"]
            }
        ]
    }

def generate_document_reference(patient_id, file_path):
    """Generate a FHIR DocumentReference pointing to the WSI."""
    return {
        "resourceType": "DocumentReference",
        "status": "current",
        "subject": {
            "reference": f"Patient/{patient_id}"
        },
        "content": [
            {
                "attachment": {
                    "url": f"file://{file_path}",
                    "title": os.path.basename(file_path),
                    "contentType": "image/tiff"
                }
            }
        ]
    }

def post_to_fhir(resource):
    """Post a resource to the FHIR server."""
    resource_type = resource["resourceType"]
    
    # If it's a patient, we can use PUT to ensure we use the specific ID
    if resource_type == "Patient":
        url = f"{FHIR_URL}/{resource_type}/{resource['id']}"
        response = requests.put(url, headers=HEADERS, json=resource)
    else:
        url = f"{FHIR_URL}/{resource_type}"
        response = requests.post(url, headers=HEADERS, json=resource)
        
    if response.status_code in [200, 201]:
        print(f"Successfully loaded {resource_type} for {resource.get('id', 'DocumentReference')}")
    else:
        print(f"Error loading {resource_type}: {response.status_code} - {response.text}")

def main():
    print(f"Starting ingestion process. Scanning {WSI_MOUNT_DIR} for SVS files...")
    
    # Find all SVS files
    svs_files = glob.glob(os.path.join(WSI_MOUNT_DIR, "*.svs"))
    
    if not svs_files:
        print(f"No SVS files found in {WSI_MOUNT_DIR}")
        return

    print(f"Found {len(svs_files)} files. Processing up to 20 for simulation...")
    
    # Regex to extract TCGA ID (e.g. TCGA-3C-AALI)
    tcga_pattern = re.compile(r"(TCGA-[A-Z0-9]{2}-[A-Z0-9]{4})")
    
    processed_patients = set()
    
    for file_path in svs_files[:20]:
        filename = os.path.basename(file_path)
        match = tcga_pattern.search(filename)
        
        if not match:
            print(f"Could not extract TCGA ID from {filename}")
            continue
            
        patient_id = match.group(1)
        
        # 1. Create Patient if not already created
        if patient_id not in processed_patients:
            patient_resource = generate_synthetic_patient(patient_id)
            post_to_fhir(patient_resource)
            processed_patients.add(patient_id)
            
        # 2. Create DocumentReference pointing to the WSI
        doc_ref_resource = generate_document_reference(patient_id, file_path)
        post_to_fhir(doc_ref_resource)

if __name__ == "__main__":
    main()
