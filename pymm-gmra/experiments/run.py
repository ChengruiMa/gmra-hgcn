import os
import subprocess
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

def email(success, input_file, output_file, error_type=None):
    # Email configurations
    sender_email = "chengrui.ma.27@dartmouth.edu"
    receiver_email = "chengrui.ma.27@dartmouth.edu"

    # Create a multipart message
    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = receiver_email

    if success:
        subject = f"{input_file} Completed Successfully"
        message = f"Output for {input_file} saved to {output_file}"
    else:
        if error_type == "MemoryError":
            subject = f"MemoryError Encountered for Script: {input_file}"
            message = f"A MemoryError occurred while running {input_file}."
        
        elif error_type == "SystemError":
            subject = f"SystemError Encountered for Script: {input_file}"
            message = f"A SystemError occurred while running {input_file}."
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
    # List all Python scripts in the directory
    python_scripts = [file for file in os.listdir(directory) if file.endswith(".py")]

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
directory_path = "/scratch/f0071gk/mcas-gmra/pymm-gmra/experiments/stellargraph_ecai_demos/Link_Prediction"

# Run scripts directly in the specified directory
run_scripts_in_directory(directory_path)
