import socket, sys, shlex, signal, json, select

from PV_Circuit_Model.data_fitting_tandem_cell import (
    get_measurements, analyze_solar_cell_measurements, generate_differentials
)

SHOULD_EXIT = False

def _sigint(_sig, _frm):
    # allow Ctrl+C to request shutdown even during blocking waits
    global SHOULD_EXIT
    SHOULD_EXIT = True

signal.signal(signal.SIGINT, _sigint)  # handle Ctrl+C

def analyze_solar_cell_measurements_wrapper(measurements_folder,sample_info,f_out):
    measurements = get_measurements(measurements_folder)
    cell_model, _ = analyze_solar_cell_measurements(measurements,sample_info=sample_info,use_fit_dashboard=True,f_out=f_out,is_tandem=sample_info["is_tandem"],silent_mode=False)
    print(cell_model)
    # need to f_out all this stuff
    if sample_info["is_tandem"]:
        output = f"OUTPUT:[{cell_model.cells[0].J01()},{cell_model.cells[0].J02()},{cell_model.cells[0].specific_shunt_cond()},{cell_model.cells[1].J01()},{cell_model.cells[1].J02()},{cell_model.cells[1].PC_J01()},{cell_model.cells[1].specific_shunt_cond()},{cell_model.specific_Rs()}]\n"
    else:
        output = f"OUTPUT:[{cell_model.J01()},{cell_model.J02()},{cell_model.specific_shunt_cond()},{cell_model.specific_Rs()}]\n"
    f_out.write(output)
    f_out.flush()
    return measurements, cell_model

def generate_differentials_wrapper(measurements,cell_model,f_out):
    M, Y, fit_parameters, aux = generate_differentials(measurements,cell_model)
    f_out.write(f"OUTPUT:{json.dumps(M.tolist())}\n")
    f_out.write(f"OUTPUT:{json.dumps(Y.tolist())}\n")
    fit_parameter_aspects = ["limit_order_of_mag","this_min","this_max","abs_min","abs_max","min","max","value","nominal_value","d_value","is_log"]
    for aspect in fit_parameter_aspects:
        f_out.write(f"OUTPUT:{fit_parameters.get(aspect)}\n")
    alpha = 1e-5 
    regularization_method=0 
    limit_order_of_mag = False
    if "alpha" in aux:
        alpha = aux["alpha"]
    if "limit_order_of_mag" in aux:
        limit_order_of_mag = aux["limit_order_of_mag"]
    if "regularization_method" in aux:
        regularization_method = aux["regularization_method"]
    f_out.write(f"OUTPUT:{alpha}\n")
    f_out.write(f"OUTPUT:{regularization_method}\n")
    f_out.write(f"OUTPUT:{limit_order_of_mag}\n")

def handle_block(lines,variables,f_out):
    #try:
        for s in lines:
            s = s.strip()
            words = shlex.split(s)
            if len(words)==0:
                continue
            command = words[0]
            match command:
                case "QUIT":
                    return "BYE"
                case "MAKESTARTINGGUESS": # e.g. MAKETANDEMSTARTINGGUESS measurements_folder wafer_area bottom_cell_thickness enble_Auger top_cell_JL bottom_cell_JL
                    if len(words)>=6:
                        measurements_folder = words[1]
                        try:
                            wafer_area = float(words[2])
                        except ValueError:
                            wafer_area = None
                        try:
                            bottom_cell_thickness = float(words[3])
                        except ValueError:
                            bottom_cell_thickness = None
                        try:
                            enable_Auger = words[4]
                        except ValueError:
                            enable_Auger = None
                        try:
                            bottom_cell_JL = float(words[5])
                        except ValueError:
                            bottom_cell_JL = None
                        if len(words)>6:
                            try:
                                top_cell_JL = float(words[6])
                            except ValueError:
                                top_cell_JL = None
                        if wafer_area is not None and bottom_cell_thickness is not None:
                            sample_info = {"area":wafer_area,"bottom_cell_thickness":bottom_cell_thickness,"enable_Auger":enable_Auger}
                            if top_cell_JL is not None and bottom_cell_JL is not None:
                                sample_info["is_tandem"] = True
                            else:
                                sample_info["is_tandem"] = False
                            variables["measurements"], variables["cell_model"] = analyze_solar_cell_measurements_wrapper(measurements_folder,sample_info,f_out)
                            if top_cell_JL is not None and bottom_cell_JL is not None:
                                variables["cell_model"].set_JL([bottom_cell_JL,top_cell_JL])
                            else:
                                variables["cell_model"].set_JL(bottom_cell_JL)
                            _, Vmp, _ = variables["cell_model"].get_Pmax(return_op_point=True)
                            f_out.write(f"OUTPUT:{Vmp}\n") # send a set point for Griddler to calculate Rs

                case "SIMULATEANDCOMPARE":
                    if "cell_model" in variables:
                        success = True
                        if len(words)==9:
                            function_calls = [
                                variables["cell_model"].cells[0].set_J01,
                                variables["cell_model"].cells[0].set_J02,
                                variables["cell_model"].cells[0].set_specific_shunt_cond,
                                variables["cell_model"].cells[1].set_J01,
                                variables["cell_model"].cells[1].set_J02,
                                variables["cell_model"].cells[1].set_PC_J01,
                                variables["cell_model"].cells[1].set_specific_shunt_cond,
                                variables["cell_model"].set_specific_Rs_cond
                            ]
                        else:
                            function_calls = [
                                variables["cell_model"].set_J01,
                                variables["cell_model"].set_J02,
                                variables["cell_model"].set_specific_shunt_cond,
                                variables["cell_model"].set_specific_Rs_cond
                            ]
                        for i in range(len(function_calls)):
                            try:
                                number = float(words[i+1])
                                function_calls[i](number)
                            except ValueError:
                                success = False
                                break
                        if success:
                            generate_differentials_wrapper(variables["measurements"],variables["cell_model"],f_out)
                case _:
                    f_out.write(f"Unknown command: {command}\n")
                    f_out.flush()
        return "FINISHED"
    # except Exception as e:
    #     return "FAILED: " + str(e)

def read_block(f_in):
    """Read lines until END (returns list[str]). Returns None if connection closed."""
    lines = []
    while True:
        try:
            line = f_in.readline()
        except socket.timeout:
            # allow main loop to notice Ctrl+C
            if SHOULD_EXIT:
                return None
            continue
        if line == "":
            return None  # client closed
        line = line.rstrip("\r\n")
        if line == "END":
            return lines
        lines.append(line)

def read_block_sock(conn):
    """Read until a line equal to END is seen or the peer closes.
       Returns a list[str] of lines (sans newlines), or None on clean close."""
    buf = b""
    while True:
        r, _, _ = select.select([conn], [], [], 1.0)
        if not r:
            if SHOULD_EXIT:
                return None
            continue

        chunk = conn.recv(4096)
        if not chunk:
            # EOF: return any partial lines we already got
            if buf:
                print("EOF with partial buffer:", repr(buf))
                text = buf.decode("utf-8", errors="replace")
                lines = [ln for ln in text.splitlines() if ln.strip().upper() != "END"]
                return lines
            print("EOF with no data")
            return None

        print("RX BYTES:", repr(chunk))  # <-- proves data is arriving
        buf += chunk

        # Did we receive END?
        if b"\r\nEND\r\n" in buf or b"\nEND\n" in buf or b"\rEND\r" in buf or b"\nEND\r\n" in buf:
            text = buf.decode("utf-8", errors="replace")
            lines = []
            for ln in text.splitlines():
                if ln.strip().upper() == "END":
                    return lines
                lines.append(ln)

def main():
    host, port = "127.0.0.1", 5007
    if len(sys.argv) >= 2:
        try:
            port = int(sys.argv[1])
        except ValueError:
            print("usage: server.py <port>")
            sys.exit(1)
    # ... parse argv ...

    print(f"Starting server on {host}:{port}")
    variables = {}

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((host, port))
        s.listen(8)               # allow a small backlog
        s.settimeout(1.0)         # so Ctrl+C is noticed


        try:
            while not SHOULD_EXIT:
                try:
                    conn, addr = s.accept()
                except socket.timeout:
                    continue

                with conn:
                    print("ACCEPTED:", addr)
                    conn.settimeout(None)  # select() handles timing; keep it blocking for recv
                    conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

                    while not SHOULD_EXIT:
                        block = read_block_sock(conn)
                        if block is None:
                            break  # client closed
                        print("BLOCK:", block)
                        # If you still want text writing via makefile, you can keep f_out:
                        f_out = conn.makefile("w", encoding="utf-8", newline="\n")
                        result = handle_block(block, variables, f_out)
                        if result == "BYE":
                            f_out.write("FINISHED\n"); f_out.flush()
                            return
                        f_out.write(result + "\n"); f_out.flush()
        except KeyboardInterrupt:
            pass
    print("Server stopped.")

if __name__ == "__main__":
    main()