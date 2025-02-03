def view_logs(log_file="test_accuracy.log"):
    try:
        with open(log_file, "r") as file:
            logs = file.readlines()
            print("---- LOG FILE CONTENT ----")
            for line in logs:
                print(line.strip())
    except FileNotFoundError:
        print(f"Log file '{log_file}' not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    view_logs()
