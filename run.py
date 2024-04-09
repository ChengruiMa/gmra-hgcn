import os
import subprocess
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

def email(success, input_file, output_file, error_type=None):
    # Email configurations
    sender_email = "felicia.schenkelberg.th@dartmouth.edu"
    receiver_email = "felicia.schenkelberg.th@dartmouth.edu"

    # Create a multipart message
    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = receiver_email

    if success:
        subject = f"{input_file} Completed Sucessfully"
        message = f"Outpout for {input_file} saved to {output_file}"
    else:
        if error_type == "MemoryError":
            subject = "MemoryError Encountered for Script: {input_file}"
            message = f"A MemoryError occurred while while running {input_file}."
        
        elif error_type == "SystemError":
            subject = "SystemError Encountered for Script: {input_file}"
            message = f"A SystemError occurred while while running {input_file}."
        else:
            subject = f"An error occurred while running {input_file}."
            message = f"{error_type}"

    msg['Subject'] = subject
    msg.attach(MIMEText(message, 'plain'))

    # Connect to the SMTP server
    with smtplib.SMTP('smtp.dartmouth.edu', 25) as server:
        # Send the email
        server.sendmail(sender_email, receiver_email, msg.as_string())

def run_scripts_in_directory(directory):
    # List all files in the directory
    files = os.listdir(directory)
    
    # Filter out only the Python scripts (assuming they have a .py extension)
    python_scripts = [file for file in files if file.endswith(".py")]

    # Loop through each Python script and try to run it using Python 3
    for script in python_scripts:
        script_path = os.path.join(directory, script)
        try:
            # Run the script using subprocess and python3
            subprocess.run(["python3", script_path], check=True)
            email(success=True, input_file=script, output_file=script_path, error_type=None)
        except subprocess.CalledProcessError as e:
            # Catch any errors that occur during script execution
            email(success=False, input_file=script, output_file=script_path, error_type=str(e))

# Specify the directory containing the scripts
directory_path = "/scratch/f006dg0/mcas-gmra/experiments/Link_prediction"

# Call the function to run the scripts in the directory
run_scripts_in_directory(directory_path)
