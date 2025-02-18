import sys

# Arbitrary example
ISO_JOBS = [
    "CZ", "SE", "FR", "GB"
]

def do_something(iso_code: str):
    print(f"ISO code: {iso_code}")

if __name__ == "__main__":

    # Retrieve the index of the batch job
    # These start with 1, so we need to adjust this for 0-indexing later
    array_task_id = int(sys.argv[2])

    print(f"Task ID: {array_task_id}")
    do_something(ISO_JOBS[array_task_id-1])