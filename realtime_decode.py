import cv2
from mevid import compute_diff


if __name__ == "__main__":
    cap = cv2.VideoCapture('datasets/blackisgood.webm')
    i = 0
    for i in range(350):
        ret, frame = cap.read()
    
    while True:
        print('decoding frame %d' % i)
        ret, frame = cap.read()
    
        try:
            cv2.imshow('orig', cv2.resize(frame, (1280, 720)))
            sol = compute_diff(frame.astype('uint8'), 265, nd=32)
            cv2.imshow('sol', sol/256)
            #plt.imshow(sol, cmap='gray') 
            #plt.show()
        except Exception as e:
            logging.exception('shit happended while decoding frame %d; skipping' % i)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        i += 1
    
    cap.release()
    cv2.destroyAllWindows()
