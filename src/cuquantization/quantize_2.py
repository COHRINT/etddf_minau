import rospy

from etddf_minau.msg import MeasurementPackage, Measurement
import numpy as np
import sys
import rospy
import struct
import time

"""
Structure of Pkg (28 bytes)
1 Byte: Delta Multiplier
1 Byte: header
1 Byte: time
(if necessary)
1 Byte: data

"""

def normalize_angle(angle):
    while angle > 360.0:
        angle -= 360.0
    while angle < 0:
        angle += 360.0
    return angle

################ API FUNCTIONS ################
def measPkg2Bytes(meas_pkg, asset_landmark_dict, max_latency, packet_size=32):
    """Converts an etddf/MeasurementPackage.msg to byte stream

    Args:
        meas_pkg (etddf/MeasurementPackage.msg): The measurement package to compress
        asset_landmark_dict (dict): Agreed upon dictionary to go from asset_name --> integer 0-15.
            Must be the same on every agent.
            e.g. {'bluerov2_4' : 0, 'bluerov2_3' : 1, 'landmark_pole1' : 2, 'red_agent' : 3}
        packet_size (int): Number of bytes to use. Unused bytes are packed to fill packet_size

    Returns:
        str: Byte stream
    """
    assert len(asset_landmark_dict) < 9 # Maximum of 8 agents
    byte_string = []

    dm = int( meas_pkg.delta_multiplier )
    assert dm > 0 and dm < 2**5
    src_id = asset_landmark_dict[meas_pkg.src_asset]
    first_header = dm << 5 | src_id
    byte_string.append(first_header)

    # Measurements
    time_per_bin = max_latency / 2**2 # 2 bits for the timestamp

    present_time = rospy.get_rostime()
    for meas in meas_pkg.measurements:
        if meas.meas_type not in HEADERS:
            raise ValueError(meas.meas_type + " not in supported types: " + str(HEADERS.keys()))
        header = HEADERS[meas.meas_type]
        header2 = asset_landmark_dict[meas.measured_asset]
        timestamp = (present_time - meas.stamp).to_sec()
        time_bin = int( timestamp / time_per_bin )
        if time_bin > 3:
            time_bin = 3

        data_bin = 0

        if meas.meas_type == "sonar_range":
            if meas.data > MEAS_MAX_VALUES[meas.meas_type]:
                raise ValueError("{} meas outside of compression bounds [0-{}]: ".format(meas.meas_type, MEAS_MAX_VALUES[meas.meas_type]) + str(meas.data))
            else:
                bin_per_meter = 255 / float(MEAS_MAX_VALUES[meas.meas_type])
                data_bin = int(meas.data * bin_per_meter)
        elif meas.meas_type == "sonar_azimuth":
            if meas.data > MEAS_MAX_VALUES[meas.meas_type]:
                raise ValueError("{} meas outside of compression bounds [0-{}]: ".format(meas.meas_type, MEAS_MAX_VALUES[meas.meas_type]) + str(meas.data))
            else:
                data = normalize_angle(meas.data)
                bin_per_meter = 255 / float(MEAS_MAX_VALUES[meas.meas_type])
                data_bin = int(data * bin_per_meter)
        elif meas.meas_type == "modem_range":
            if meas.data > MEAS_MAX_VALUES[meas.meas_type]:
                raise ValueError("{} meas outside of compression bounds [0-{}]: ".format(meas.meas_type, MEAS_MAX_VALUES[meas.meas_type]) + str(meas.data))
            else:
                bin_per_meter = 255 / float(MEAS_MAX_VALUES[meas.meas_type])
                data_bin = int(meas.data * bin_per_meter)
        elif meas.meas_type == "modem_azimuth":
            if meas.data > MEAS_MAX_VALUES[meas.meas_type]:
                raise ValueError("{} meas outside of compression bounds [0-{}]: ".format(meas.meas_type, MEAS_MAX_VALUES[meas.meas_type]) + str(meas.data))
            else:
                data = normalize_angle(meas.data)
                bin_per_meter = 255 / float(MEAS_MAX_VALUES[meas.meas_type])
                data_bin = int(data * bin_per_meter)
        elif "implicit" in meas.meas_type:
            data_bin = None

        # Append to bytestring
        full_header = header << 5 | header2 << 2 | time_bin
        byte_string.append(full_header)
        # byte_string.append(timestamp)
        if data_bin is not None:
            byte_string.append(data_bin)

    print("Packet Cost: {}".format(len(byte_string)))

    if len(byte_string) > packet_size:
        raise ValueError("Compression failed. Byte string {} is greater than packet size {} with meas count {}".format(len(byte_string), packet_size, len(meas_pkg.measurements)))

    # Pack empty values to the end of the buffer for unused space
    byte_string.extend([HEADERS['empty'] for x in range(packet_size - len(byte_string))])

    # Map all values -128 to 127
    byte_string = [x - 128 for x in byte_string]
    return byte_string

def bytes2MeasPkg(byte_arr, transmission_time, asset_landmark_dict, max_latency, global_pose):
    """Converts a byte stream compressed using measPkg2Bytes() to a Measurement Package

    Args:
        byte_arr (str): byte stream to decompress
        transmission_time (int): Estimated time delta between when measPkg2Bytes() was called
            and this method has been called
            If unknown set to 0; Not critical to be accurate
        asset_landmark_dict (dict): Agreed upon dictionary to go from asset_name --> integer 0-15.
            Must be the same on every agent.
            e.g. {'bluerov2_4' : 0, 'bluerov2_3' : 1, 'landmark_pole1' : 2, 'red_agent' : 3}
        global_pose (list): Pose of the surface beacon
            e.g. [x,y,z,theta]

    Returns:
        etddf/MeasurementPackage.msg: Measurement Package
    """

    # Map all values 0 to 255
    byte_arr = [x + 128 for x in byte_arr]

    mp = MeasurementPackage()

    primary_header = byte_arr[0]
    src_asset_code = primary_header & 7
    mp.src_asset = asset_landmark_dict.keys()[asset_landmark_dict.values().index( src_asset_code )]
    mp.delta_multiplier = primary_header >> 3

    time_per_bin = max_latency / 2**2 # 2 bits for the timestamp

    index = 1
    present_time = rospy.get_rostime()
    while index < len(byte_arr):
        # msg_global_pose = global_pose
        header = byte_arr[index]
        # full_header = header << 5 | header2 << 2 | time_bin
        main_header = header >> 5
        header2 = (header & 0x1c) >> 2
        time_bin = header & 3
        meas_type = HEADERS.keys()[HEADERS.values().index( main_header )]
        if meas_type == "empty":
            break
        measured_agent = asset_landmark_dict.keys()[asset_landmark_dict.values().index( header2 )]
        past_time = time_bin * time_per_bin
        timestamp = rospy.Time( (present_time.secs - transmission_time) - past_time )

        data = 0
        if "implicit" not in meas_type:
            index += 1

            data_bin = byte_arr[index]

            if meas_type == "sonar_range":
                unit_per_bin = float(MEAS_MAX_VALUES[meas_type]) / 255.0
                data = data_bin * unit_per_bin
            elif meas_type == "sonar_azimuth":
                unit_per_bin = float(MEAS_MAX_VALUES[meas_type]) / 255.0
                data = data_bin * unit_per_bin
                data = np.mod( data + 180, 360) - 180 # -180 to 180
            elif meas_type == "modem_range":
                unit_per_bin = float(MEAS_MAX_VALUES[meas_type]) / 255.0
                data = data_bin * unit_per_bin
            elif meas_type == "modem_azimuth":
                unit_per_bin = float(MEAS_MAX_VALUES[meas_type]) / 255.0
                data = data_bin * unit_per_bin
                data = np.mod( data + 180, 360) - 180 # -180 to 180
        
        m = Measurement(meas_type, timestamp, mp.src_asset, measured_agent, data, 0.0, [], 0.0)
        mp.measurements.append(m)
        index += 1

    return mp

################ PRIVATE MEMBERS ################

HEADERS = {
    "empty" : 0,
    "sonar_range" : 1,
    "sonar_range_implicit" : 2,
    "sonar_azimuth" : 3,
    "sonar_azimuth_implicit" : 4,
    "modem_range" : 5,
    "modem_azimuth" : 6
}
MEAS_MAX_VALUES = {
    "sonar_range" : 20.0,
    "sonar_azimuth" : 360.0, 
    "modem_range" : 50.0,
    "modem_azimuth" : 360.0 
}

if __name__ == "__main__":
    # Create measurement package
    rospy.init_node("test_quantize")

    global_pose = [1,2,3,4]

    asset_landmark_dict = {"dory" : 0, 
        "guppy" : 1, "surface" : 2
    }

    print('############ TEST 1 #################')
    # Test compression and decompression
    mp = MeasurementPackage()
    mp.src_asset = "surface"
    mp.delta_multiplier = 1.0
    t = rospy.get_rostime()
    # m = Measurement("modem_range", t, mp.src_asset, "dory", 3.65, 0.5, global_pose, 0.0)
    # m2 = Measurement("modem_azimuth", t, mp.src_asset, "dory", -65.72, 0.5, global_pose, 0.0)
    m3 = Measurement("modem_range", t-rospy.Duration(5), mp.src_asset, "guppy", 7.8, 0.5, global_pose, 0.0)
    # m4 = Measurement("modem_azimuth", t-rospy.Duration(5), mp.src_asset, "guppy", 23.0, 0.5, global_pose, 0.0)
    # mp.measurements.append(m)
    # mp.measurements.append(m2)
    mp.measurements.append(m3)
    # mp.measurements.append(m4)

    
    num_bytes_buffer = 29
    max_latency = 16
    bytes_ = measPkg2Bytes(mp, asset_landmark_dict, max_latency, num_bytes_buffer)
    mp_return = bytes2MeasPkg(bytes_, 0, asset_landmark_dict, max_latency, global_pose)
    for i in range(len(mp.measurements)):
        print(mp.measurements[i])
        print(mp_return.measurements[i])
        print("####")

    # print('############ TEST 2 #################')
    # mp = MeasurementPackage()
    # mp.src_asset = "dory"
    # mp.delta_multiplier = 5.0

    # t = rospy.get_rostime()
    # m = Measurement("sonar_x_burst", t, mp.src_asset, "guppy", 5.24, 0.5, [], 1)
    # m1 = Measurement("sonar_x", t, mp.src_asset, "guppy", 2.1, 0.5, [], 1)
    # m2 = Measurement("sonar_y_burst", t, mp.src_asset, "guppy", 6.3, 0.5, [], 1)
    # m3 = Measurement("sonar_y", t, mp.src_asset, "guppy", -6.5, 0.5, [], 1)
    # rospy.sleep(2)
    # t = rospy.get_rostime()
    # m4 = Measurement("sonar_x", t, mp.src_asset, "guppy", 2.1, 0.5, [], 1)
    # m5 = Measurement("sonar_y", t, mp.src_asset, "guppy", -6.5, 0.5, [], 1)

    # mp.measurements.append(m)
    # mp.measurements.append(m1)
    # mp.measurements.append(m2)
    # mp.measurements.append(m3)
    # mp.measurements.append(m4)
    # mp.measurements.append(m5)
    # print(mp)
    # bytes_ = measPkg2Bytes(mp, asset_landmark_dict, num_bytes_buffer)
    # mp_return = bytes2MeasPkg(bytes_, 2, asset_landmark_dict, global_pose)
    # print(mp_return)

