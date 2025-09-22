mkdir -p checkpoint

mkdir -p checkpoint/sam2
mkdir -p checkpoint/dover
mkdir -p checkpoint/lpips
mkdir -p checkpoint/video_depth_anything


wget https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt \
    -O checkpoint/sam2/sam2_hiera_large.pt
wget https://github.com/QualityAssessment/DOVER/releases/download/v0.1.0/DOVER.pth \
    -O checkpoint/DOVER/DOVER.pth
wget wget https://download.pytorch.org/models/vgg16-397923af.pth \
    -O checkpoint/lpips/vgg.pth
wget https://huggingface.co/depth-anything/Video-Depth-Anything-Small/resolve/main/video_depth_anything_vits.pth?download=true \
    -O checkpoint/video_depth_anything/video_depth_anything_vits.pth
huggingface-cli download IDEA-Research/grounding-dino-tiny \
    --local-dir checkpoint/IDEA-Research/grounding-dino-tiny
