import requests
import time
from Bio import Entrez
from Bio import SeqIO
import os
import pandas as pd
import re
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())

Entrez.email = os.environ.get('USER_EMAIL')

def get_ensembl_ids(uniprot_ids):
    # Define the URL for the ID mapping
    url = 'https://rest.uniprot.org/idmapping/run'
    
    # Create the payload for the POST request
    payload = {
        'ids': ','.join(uniprot_ids),
        'from': 'UniProtKB_AC-ID',
        'to': 'Ensembl'
    }
    
    # Make the POST request
    response = requests.post(url, data=payload)

    # Check for a successful request
    if response.status_code != 200:
        print(f"Error: {response.status_code} - {response.text}")
        return None

    # Extract the mapping job ID from the response
    job_id = response.json().get('jobId')
    
    # Poll for the result until it's ready
    while True:
        # Check the status of the job
        status_url = f'https://rest.uniprot.org/idmapping/status/{job_id}'
        status_response = requests.get(status_url)

        if status_response.status_code != 200:
            print(f"Error checking status: {status_response.status_code} - {status_response.text}")
            return None
        
        status = status_response.json().get('status')
        
        # If the status is 'FINISHED', retrieve the results
        if status == 'FINISHED':
            result_url = f'https://rest.uniprot.org/idmapping/results/{job_id}'
            result_response = requests.get(result_url)
            
            if result_response.status_code != 200:
                print(f"Error retrieving results: {result_response.status_code} - {result_response.text}")
                return None
            
            # Return the results as a list of mappings
            return result_response.json().get('results')
        
        # If the job is still running, wait and check again
        elif status == 'RUNNING':
            print("Job is still running, checking again in 5 seconds...")
            time.sleep(5)
        
        # If the job has failed, exit
        elif status == 'FAILED':
            print("Job has failed.")
            return None

    
def parse_uniprot_dat(file_path):
    uniprot_to_ensembl = {}

    with open(file_path, 'r') as file:
        lines = file.readlines()

        current_uniprot_id = None

        for line in lines:
            # Normalize line spacing and avoid processing empty lines
            line = line.strip()
            if not line:
                continue
                
            # Split the line by tabs
            parts = line.split("\t")  # Use tab as the delimiter
            
            # Check if the line has at least 3 parts
            if len(parts) < 3:
                continue
            
            # Extract UniProt ID and the ID type (e.g., Ensembl, UniProtKB-ID)
            uniprot_id = parts[0]
            id_type = parts[1]
            id_value = parts[2].strip()  # The value of the ID
            
            # If the ID is UniProtKB-ID, we just set the current UniProt ID
            if id_type == "UniProtKB-ID":
                if uniprot_id not in uniprot_to_ensembl:
                    uniprot_to_ensembl[uniprot_id] = []
            
            # If it's an Ensembl ID, add it to the list for the current UniProt ID
            elif id_type == "Ensembl":
                if uniprot_id in uniprot_to_ensembl:
                    uniprot_to_ensembl[uniprot_id].append(id_value)

    return uniprot_to_ensembl


def get_genes_in_region_ensembl(chromosome, start, end, species="human"):
    """Get genes in a specified genomic region using Ensembl REST API."""
    species_map = {
        "human": "homo_sapiens",
        "mouse": "mus_musculus"
    }
    
    ensembl_species = species_map.get(species.lower())
    if not ensembl_species:
        raise ValueError(f"Unsupported species: {species}")

    server = "https://rest.ensembl.org"
    ext = f"/overlap/region/{ensembl_species}/{chromosome}:{start}-{end}?feature=gene"
    
    headers = {"Content-Type": "application/json"}
    response = requests.get(server + ext, headers=headers)

    if not response.ok:
        response.raise_for_status()

    genes = []
    for gene in response.json():
        genes.append({
            "ensembl_id": gene.get("id"),
            "symbol": gene.get("external_name"),
            "description": gene.get("description"),
            "start": gene.get("start"),
            "end": gene.get("end"),
            "strand": gene.get("strand")
        })
    
    return genes

def symbol_to_ensembl_id(symbol, species="human"):
    """Convert gene symbol to Ensembl gene ID."""
    species_map = {
        "human": "homo_sapiens",
        "mouse": "mus_musculus"
    }
    ensembl_species = species_map.get(species.lower())
    if not ensembl_species:
        raise ValueError(f"Unsupported species: {species}")
    
    url = f"https://rest.ensembl.org/xrefs/symbol/{ensembl_species}/{symbol}?object_type=gene"
    headers = {"Content-Type": "application/json"}
    r = requests.get(url, headers=headers)
    
    if not r.ok:
        r.raise_for_status()

    data = r.json()
    if data:
        return data[0]["id"]  # Ensembl gene ID
    return None

def ensembl_to_uniprot(ensembl_id):
    """Convert Ensembl gene ID to UniProt ID(s)."""
    url = f"https://rest.ensembl.org/xrefs/id/{ensembl_id}?external_db=UniProtKB/Swiss-Prot"
    headers = {"Content-Type": "application/json"}
    r = requests.get(url, headers=headers)
    
    if not r.ok:
        r.raise_for_status()

    return [entry["primary_id"] for entry in r.json()]

def uniprot_to_ensembl(uniprot_id):
    """Convert UniProt ID to Ensembl gene ID(s)."""
    url = f"https://rest.ensembl.org/xrefs/symbol/homo_sapiens/{uniprot_id}?external_db=UniProtKB/Swiss-Prot"
    headers = {"Content-Type": "application/json"}
    r = requests.get(url, headers=headers)
    
    if not r.ok:
        r.raise_for_status()

    return [entry["id"] for entry in r.json()]


def convert_genomic_location_to_ensembl_ids(gene_string):
    """
    Convert a gene string containing genomic locations and/or gene symbols to Ensembl IDs.
    
    Args:
        gene_string (str): String containing gene info, possibly in format "chr14:69216270-69216293"
                          or as gene symbols, separated by "_"
        utils: Object containing get_genes_in_region_ensembl and symbol_to_ensembl_id methods
    
    Returns:
        list: List of Ensembl IDs corresponding to the genes in the input string
    """
    # If gene_string is NaN or None, return empty list
    if pd.isna(gene_string):
        return []
    
    pattern = r'^(chr[0-9XYM]+):(\d+)-(\d+)$'
    gene_list = gene_string.split('_')
    print(f'{gene_string=}')
    print(f'{gene_list=}')
    
    ensembl_ids = []
    
    for gene in gene_list:
        match = re.match(pattern, gene)
        print(f'{match=}')
        
        if match:      # If genomic location format
            chromosome = match.group(1)
            start = int(match.group(2))
            end = int(match.group(3))
            genes_at_location = get_genes_in_region_ensembl(chromosome, start, end)
            
            # Extract Ensembl IDs from genes at this location
            location_ensembl_ids = [gene_info['ensembl_id'] for gene_info in genes_at_location]
            ensembl_ids.extend(location_ensembl_ids)
        elif gene.startswith('ENSG'):       # If Ensembl format
            ensembl_ids.append(gene)
        elif gene == 'control':             # If control
            ensembl_ids.append('control')
        else:                               # If gene symbol format
            ensembl_id = symbol_to_ensembl_id(gene)
            if not ensembl_id:
                ensembl_id = 'None'
            ensembl_ids.append(ensembl_id)
    
    ensembl_ids = '_'.join(ensembl_ids)
    print(f'{gene_list}:     {ensembl_ids}')
    return ensembl_ids