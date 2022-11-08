import torch
import pycocotools.mask as mask_util
import numpy as np

def create_image(width, height, left_fill, right_fill, top_fill, bottom_fill, interior_fill):
    image = torch.zeros(size=[height, width], device='cpu', dtype=torch.uint8)
    if left_fill:
        image[:,0:1] = 1
    if right_fill:
        image[:,width-1:] = 1
    if top_fill:
        image[0:1,:] = 1
    if bottom_fill:
        image[height-1:,:] = 1
    if interior_fill:
        image[1:height-1,1:width-1] = 1
    return image

def pad_image(image, left_pad, right_pad, top_pad, bottom_pad):
    if left_pad == 0 and right_pad == 0 and top_pad == 0 and bottom_pad == 0:
        return image
    else:
        height, width = image.shape
        padded_image = torch.zeros(size=[height+top_pad+bottom_pad, width+left_pad+right_pad], device=image.device, dtype=image.dtype)
        padded_image[top_pad:top_pad+height, left_pad:left_pad+width].copy_(image)
        return padded_image

def bools(arg):
    return "T" if arg else "F"

def check_image(n, width, height, left_fill, right_fill, top_fill, bottom_fill, interior_fill, left_pad, right_pad, top_pad, bottom_pad, silent_on_success=True, print_raw_rles=False):
    image = create_image(width, height, left_fill, right_fill, top_fill, bottom_fill, interior_fill)
    padded_image = pad_image(image, left_pad, right_pad, top_pad, bottom_pad)
    padded_height, padded_width = padded_image.shape
    c = np.array(image[ :, :, np.newaxis], order="F")
    pad_c = np.array(padded_image[ :, :, np.newaxis], order="F")
    rle = mask_util.encode(c, paste_args=dict(oy=top_pad, ox=left_pad, oh=padded_height, ow=padded_width))
    rle_ref = mask_util.encode(pad_c)
    success = True if rle == rle_ref else False
    if not silent_on_success or not success:
        if print_raw_rles:
            print()
            print(rle)
            print(rle_ref)
        print("%s :: Test %d :: w,h,lf,rf,tf,bf,if = %d,%d,%s,%s,%s,%s,%s ;; lp,rp,tp,bp = %d,%d,%d,%d" % (
            "Success" if success else "FAIL", n,
            width, height, bools(left_fill), bools(right_fill), bools(top_fill), bools(bottom_fill), bools(interior_fill), 
            left_pad, right_pad, top_pad, bottom_pad))
    return success

def call_check_image(width, height, n, silent_on_success, print_raw_rles):
    args = []
    args.append( n )
    args.append( width )
    args.append( height )
    args.append( True if n & 256 else False ) 
    args.append( True if n & 128 else False ) 
    args.append( True if n & 64 else False ) 
    args.append( True if n & 32 else False ) 
    args.append( True if n & 16 else False )
    args.append( 1 if n & 8 else 0 )
    args.append( 1 if n & 4 else 0 )
    args.append( 1 if n & 2 else 0 )
    args.append( 1 if n & 1 else 0 )
    return check_image(*args, silent_on_success=silent_on_success, print_raw_rles=print_raw_rles)

def main():
    num_success, num_total = 0,0
    for n in range(512):
        num_total += 1
        if call_check_image(100, 100, n, True, False):
            num_success += 1
    print("%d / %d Tests succeeded" % (num_success,num_total))
    #call_check_image(100, 100, 96, False, True)

if __name__ == "__main__":
    main()

