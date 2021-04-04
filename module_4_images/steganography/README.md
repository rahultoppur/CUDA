# Image Steganography
## Introduction
You are a spy working for your nation's government. You are currently operating undercover in enemy territory, and need to find a way to communicate to your HQ back home. Sending letters and emails are too risky, so you decide on something else--image steganography. You decide that you will "hide" your message (encoded as bits) within the pixels of the image that you send.

## Methodology
Encryption
* Encode the message as `N` bits (every 8 bits represents an ASCII character)
* The `R` value of the first `N` pixels of the image will contain the secret message as the **least-significant** bit (right-most bit)

Decryption
* Retrieve the first `N` pixels from the image, noting down the last bit of the `R` value of each pixel
* Convert the bits back to ASCII

## Task 1: Find the Hidden Message
Armed with your new CUDA image processing skills, you now have the ability to hide your messages within images. You get the following message from a colleague in an email along with the following photo:
```txt
Hi!

I went on a really great hike the other day and took this photo.

There was some beautiful scenery, and we were outside for 177 minutes.

Best,
 X.

---
Attachment(1)-hike_sunset.ppm
---
```
>![Safari](../../media/safari.png)

You find it odd that your friend mentioned such a specific time frame. Using `177` as the length of the encoded message (in bits), write `steg_decrypt.cu` which outputs the hidden message from `hike_sunset.ppm`.

Some starter code for the kernel has been provided for you below.
```c
/* Decode the secret message embedded in our image */
__global__ void decode(unsigned char* pixel_array, int width, int height, char* msg) {
    /* Determine the (x,y) coordinate within our image */
    ...
    /* Obtain the last bit from the R value of each pixel */
    ...
    /* Write each portion of the secret message to msg */
    ...
}
```
Your program should output a string of bytes. Find the corresponding ASCII by using a tool like [this](https://www.rapidtables.com/convert/number/index.html). Be sure to convert to **ASCII** instead of **ASCII/UTF-8**.

> What was the hidden message?\
**TODO: Your answer here**

## Hints
* Your implementation from Part 1 of this assignment will help with this section.

## Task 2: Encrypting your own Message
You want to reply to your friend's message. With a picture of your choosing, encrypt the following message within your image:
```txt
msg:  bring me a shrubbery. one that looks nice and not too expensive.
```
Use this [tool](https://www.rapidtables.com/convert/number/index.html) to get the representation in bits. Implement `steg_encrypt.cu`, which encrypts `msg` into the image you selected. Name your kernel `encrypt`.

Save your original image as `cool_pic_orig.ppm`. Save your encrypted message as `cool_pic_enc.ppm`. Push both to the current directory.

> Look at your pictures side by side. Can you spot any differences? **Why** do you think this is the case?\
**TODO: Your answer here**

You can ensure that you implemented Task 2 correctly by using the decoder you implemented in Task 1 to check your work. If you were able to decode the message, you were successful!

## Tasks
* Implement `steg-encrypt.cu` and `steg-decrypt.cu`
* Answer the **short answer questions** in Task 1 and Task 2