import serial
import time
 
receiverNum = "+84787136327"
sim800l = serial.Serial(
    port='/dev/ttyACM0',
    baudrate = 9600,
    parity=serial.PARITY_NONE,
    stopbits=serial.STOPBITS_ONE,
    bytesize=serial.EIGHTBITS,
    timeout=1
)
 
sms = "1"
time.sleep(1)
sim800l.write('AT+CMGF=1'.encode('utf-8'))
print(sim800l.readlines())
time.sleep(1)
cmd1 = "AT+CMGS=\" "+str(receiverNum)+"\"\n"
sim800l.write(cmd1.encode())
print(sim800l.read(24))
time.sleep(1)
sim800l.write(sms.encode())
sim800l.write(chr(26).encode())
print(sim800l.read(24))
