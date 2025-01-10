use crossterm::style::{Print, ResetColor, SetBackgroundColor, SetForegroundColor};
use image::Pixel;
use palette::{FromColor, Lab, Srgb};

fn compute_histogram(image: &image::GrayImage) -> [u32; 256] {
    let mut hist = [0u32; 256];
    for pixel in image.pixels() {
        let value = pixel.0[0];
        hist[value as usize] += 1;
    }
    hist
}

fn otsu_threshold(hist: &[u32; 256]) -> u8 {
    let total_pixels = hist.iter().sum::<u32>() as f64;
    let mut cumulative_sum = [0u32; 256];
    cumulative_sum[0] = hist[0];
    for i in 1..256 {
        cumulative_sum[i] = cumulative_sum[i - 1] + hist[i];
    }
    let sum_total: u32 = hist.iter().enumerate().map(|(i, &v)| i as u32 * v).sum();
    let mut max_sigma = 0.0;
    let mut threshold = 0u8;
    for t in 1u8..255 {
        let w0 = cumulative_sum[usize::from(t)] as f64 / total_pixels;
        let w1 = (cumulative_sum[255] - cumulative_sum[usize::from(t)]) as f64 / total_pixels;
        if w0 == 0.0 || w1 == 0.0 {
            continue;
        }
        let sum_w0: u32 = hist[..usize::from(t)]
            .iter()
            .enumerate()
            .map(|(i, &v)| i as u32 * v)
            .sum();
        let u0 = sum_w0 as f64 / w0;
        let sum_w1 = sum_total - sum_w0;
        let u1 = sum_w1 as f64 / w1;
        let sigma = w0 * w1 * (u1 - u0).powi(2);
        if sigma > max_sigma {
            max_sigma = sigma;
            threshold = t;
        }
    }

    // imageproc::edges::canny panics without this
    if threshold == 0 { 1 } else { threshold }
}

fn otsu_thresholding(image: &image::GrayImage) -> (f32, f32) {
    let hist = compute_histogram(image);
    let th2 = f32::from(otsu_threshold(&hist)) * 0.4;
    let th1 = th2 / 2.;
    (th1, th2)
}

fn get_image_edge_overlay(img: &image::DynamicImage) -> image::RgbaImage {
    let gray_img: image::GrayImage = img.to_luma8();

    let (canny_low, canny_high) = otsu_thresholding(&gray_img);

    let edges = imageproc::edges::canny(&gray_img, canny_low, canny_high);
    let overlay = image::RgbaImage::from_fn(edges.width(), edges.height(), |x, y| {
        let pixel = edges.get_pixel(x, y).to_owned();
        let value = pixel.channels()[0];
        let black = image::Rgba([0, 0, 0, 255]);
        let transparent = image::Rgba([0, 0, 0, 0]);
        match value {
            255 => black,
            _ => transparent,
        }
    });
    overlay
}

fn main() {
    let image_path = std::env::args().skip(1).next().unwrap();

    let (columns, rows) = crossterm::terminal::size().unwrap();
    let columns = u32::from(columns);
    let rows = u32::from(rows) - 2;

    let image = image::open(image_path).unwrap();
    let image = image.into_rgb8();
    let (width, height) = image.dimensions();

    let (columns, rows) = if rows * 2 * width < columns * height {
        (rows * 2 * width / height, rows)
    } else {
        (columns, columns * height / width / 2)
    };

    let unit_width = width / columns;
    let unit_height = height / (rows * 2);

    let lab_buffer: Vec<Lab<_, f64>> =
        palette::cast::from_component_slice::<Srgb<u8>>(image.as_raw())
            .iter()
            .map(|&c| Lab::from_color(c.into_format()))
            .collect();
    let k = columns * rows * 2;
    let m = 10;
    let labels = simple_clustering::slic(k, m, width, height, None, &lab_buffer).unwrap();
    let mut mean_color_image = image.to_vec();
    simple_clustering::image::mean_colors(&mut mean_color_image, k as usize, &labels, &lab_buffer)
        .unwrap();
    let mean_color_image_with_edges =
        image::RgbImage::from_raw(width, height, mean_color_image).unwrap();

    let result: image::RgbImage = image::ImageBuffer::from_fn(columns, 2 * rows, |x, y| {
        let x = unit_width / 2 + x * width / columns;
        let y = unit_height / 2 + y * height / (rows * 2);
        mean_color_image_with_edges.get_pixel(x, y).to_owned()
    });

    let mut stdout = std::io::stdout();

    for row in (0..2 * rows).step_by(2) {
        for col in 0..columns {
            let background = result.get_pixel(col, row).to_rgb();
            let foreground = result.get_pixel(col, row + 1).to_rgb();

            let [r, g, b] = background.0;
            let background = crossterm::style::Color::Rgb { r, g, b };
            let [r, g, b] = foreground.0;
            let foreground = crossterm::style::Color::Rgb { r, g, b };

            crossterm::execute!(
                stdout,
                SetForegroundColor(foreground),
                SetBackgroundColor(background),
                Print("▄".to_owned())
            )
            .unwrap();
        }
        crossterm::execute!(stdout, ResetColor).unwrap();
        println!("");
    }
}
