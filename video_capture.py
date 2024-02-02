import cv2

# Create a VideoCapture object
cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280) # set Width
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 960) # set Height

fourcc = cv2.VideoWriter_fourcc(*'X264')

out = cv2.VideoWriter('output.avi', fourcc, 30.0, (1280, 960))
while(cap.isOpened()):
    # Capture frame-by-frame
    ret, frame = cap.read()

    if ret == True:
        # Write the frame into the file 'output.avi'
        out.write(frame)
        # Display the resulting frame
        cv2.imshow('frame', frame)

    # Press q to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

cap.release()
out.release()
cv2.destroyAllWindows()


