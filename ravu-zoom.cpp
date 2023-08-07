// SPDX-License-Identifier: BSD-3-Clause
//
// Copyright (C) 2019 Bin Jin. All Rights Reserved.

#include <cassert>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cinttypes>

#include <algorithm>
#include <string>
#include <vector>
#include <sstream>
#include <iomanip>

#define BMP_COMPAT
#include "bmp.h"

#define DIE(...) {fprintf(stderr, __VA_ARGS__);exit(-1);}
#define PRINT(...) {fprintf(stderr, __VA_ARGS__);}

const double eps = FLT_EPSILON;

typedef std::vector<double> vector;
typedef std::vector<vector> matrix;

#define AR 1
const int radius = 2;
const int gradient_radius = std::max(radius - 1, 2);

const int quant_angle = 24;
const int quant_strength = 4;
const int quant_coherence = 3;
const int lut_size = 9;

const double min_strength[quant_strength - 1] = {0.004, 0.016, 0.05};
const double min_coherence[quant_coherence - 1] = {0.25, 0.5};

struct hashkey {
    int angle;
    int strength;
    int coherence;

    hashkey(int angle, int strength, int coherence) : angle(angle), strength(strength), coherence(coherence) { }
};

struct point_in_pixel {
    double x, y;

    union {
        double val;
        int origin[2];
    };

    point_in_pixel() {}

    point_in_pixel(double x, double y, double val) : x(x), y(y), val(val) {}
    point_in_pixel(double x, double y, int ox, int oy) : x(x), y(y) {
        origin[0] = ox;
        origin[1] = oy;
    }
};


#define POW3(x) ((x) <= 0 ? 0 : (x) * (x) * (x))
static double bicubic(double x)
{
    return (1.0/6.0) * (      POW3(x + 2)
                        - 4 * POW3(x + 1)
                        + 6 * POW3(x)
                        - 4 * POW3(x - 1));
}

const int VAR_NOT_INIT = -1;
const int VAR_CONSTANT_0 = -2;
const int VAR_CONSTANT_1 = -3;

int feature_count;
int feature_mapping[lut_size][lut_size][radius * 2][radius * 2];

void prepare_feature_mapping() {
    memset(feature_mapping, VAR_NOT_INIT, sizeof(feature_mapping));
    for (int x = 0; x < lut_size; x++) {
        if (x > 0 && x < lut_size - 1)
            continue;
        for (int y = 0; y < lut_size; y++) {
            if (y > 0 && y < lut_size - 1)
                continue;
            for (int i = 0; i < radius * 2; i++) {
                for (int j = 0; j < radius * 2; j++) {
                    feature_mapping[x][y][i][j] = VAR_CONSTANT_0;
                }
            }
        }
    }
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            feature_mapping[(lut_size - 1) * i][(lut_size - 1) * j][radius - 1 + i][radius - 1 + j] = VAR_CONSTANT_1;
        }
    }
    feature_count = 0;
    for (int x = 0; x < lut_size; x++) {
        for (int y = 0; y < lut_size; y++) {
            for (int i = 0; i < radius * 2; i++) {
                for (int j = 0; j < radius * 2; j++) {
                    int &var_now = feature_mapping[x][y][i][j];
                    if (var_now == VAR_NOT_INIT) {
                        int &var_op = feature_mapping[lut_size - 1 - x][lut_size - 1 - y][radius * 2 - 1 - i][radius * 2 - 1 - j];
                        assert(var_op == VAR_NOT_INIT);
                        var_now = var_op = feature_count ++;
                    }
                }
            }
        }
    }
}

std::string to_string(double x)
{
    std::ostringstream oss;
    oss << std::setprecision(std::numeric_limits<double>::max_digits10 + 1) << x;
    return oss.str();
}

std::string to_string(int x)
{
    std::ostringstream oss;
    oss << x;
    return oss.str();
}

template<typename T>
std::string to_string(std::vector<T> x)
{
    std::string ret = "[";
    for (int i = 0; i < (int)x.size(); i++) {
        if (i > 0)
            ret += ',';
        ret += to_string(x[i]);
    }
    ret += ']';
    return ret;
}

std::string remove_suffix(const std::string &file_name, const std::string &suffix)
{
    if (file_name.size() > suffix.size() &&
            file_name.substr(file_name.size() - suffix.size()) == suffix) {
        return file_name.substr(0, file_name.size() - suffix.size());
    }
    return file_name;
}

matrix read_image(const std::string &file_name)
{
    // for i in *.jpg; do convert -depth 16 -filter Mitchell -resize 75% "$i" -colorspace gray "$(basename "$i" .jpg).pnm"; done
    // for i in *.png; do convert -depth 16 "$i" -colorspace gray "$(basename "$i" .png).pnm"; done
    FILE* image_file = fopen(file_name.c_str(), "rb");
    if (!image_file) {
        DIE("file '%s' not found\n", file_name.c_str());
    }
    char buf[1024];
    fgets(buf, sizeof(buf), image_file);
    if (strncmp(buf, "P5", 2) != 0) {
        DIE("file '%s' is not a valid PGM file (binary)\n", file_name.c_str());
    }
    fgets(buf, sizeof(buf), image_file);
    int width, height, max_value;
    if (sscanf(buf, "%d%d", &width, &height) != 2) {
        DIE("file '%s' is not a valid PGM file (binary)\n", file_name.c_str());
    }
    fgets(buf, sizeof(buf), image_file);
    if (sscanf(buf, "%d", &max_value) != 1 || max_value != 65535) {
        DIE("file '%s' is not a valid PGM file (binary)\n", file_name.c_str());
    }

    matrix ret(height, vector(width));

    uint16_t line[width];

    for (int i = 0; i < height; i++) {
        if ((int)fread(line, sizeof(*line), width, image_file) != width) {
            DIE("file '%s' is not a valid PGM file (binary)\n", file_name.c_str());
        }
        for (int j = 0; j < width; j++) {
            line[j] = (line[j] & 255) << 8 | line[j] >> 8;
            ret[i][j] = (double)line[j] / max_value;
        }
    }
    fclose(image_file);
    return ret;
}

void write_image(const std::string &file_name, const matrix &image)
{
    int width = image[0].size(), height = image.size();
    const int MAXV = 65535;
    FILE* image_file = fopen(file_name.c_str(), "wb");
    if (!image_file) {
        DIE("unable to write to '%s'\n", file_name.c_str());
    }
    fprintf(image_file, "P5\n%d %d\n%d\n", width, height, MAXV);
    uint16_t line[width];
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            int val = std::max(0, std::min(MAXV, (int)round(MAXV * image[i][j])));
            line[j] = (val & 255) << 8 | val >> 8;
        }
        fwrite(line, sizeof(*line), width, image_file);
    }
    fclose(image_file);
}

double psnr(const matrix &a, const matrix &b)
{
    assert(a.size() == b.size() && a[0].size() == b[0].size());
    double sumsq = 0;
    int num = 0;
    for (int i = 0; i < (int)a.size(); i++) {
        for (int j = 0; j < (int)a[0].size(); j++) {
            double diff = a[i][j] - b[i][j];
            sumsq += diff * diff;
            num ++;
        }
    }
    return 10 * log10(1.0 / (sumsq / num));
}

void resample_image(const matrix &image, int target_width, int target_height, matrix *res)
{
    int width = image[0].size(), height = image.size();
    int new_width = width, new_height = height;
    bool sep_x;
    if (height != target_height) {
        new_height = target_height;
        sep_x = true;
    } else if (width != target_width) {
        new_width = target_width;
        sep_x = false;
    } else {
        *res = image;
        return;
    }
    matrix sampled(new_height, vector(new_width));
    for (int x = 0; x < new_height; x++) {
        for (int y = 0; y < new_width; y++) {
            double coord;
            if (sep_x) {
                coord = (double)(x + 0.5) / new_height * height - 0.5;
            } else {
                coord = (double)(y + 0.5) / new_width * width - 0.5;
            }
            int base = (int)floor(coord);
            double fract = coord - base;
            double w[4];
            double wsum = 0.0;
            for (int i = 0; i < 4; i++) {
                w[i] = bicubic(i - 1 - fract);
                wsum += w[i];
            }
            // assert(abs(wsum - 1.0) < eps);
            double sum = 0.0;
            for (int i = 0; i < 4; i++) {
                int sx = x, sy = y;
                if (sep_x) {
                    sx = std::max(0, std::min(height - 1, base - 1 + i));
                } else {
                    sy = std::max(0, std::min(width - 1, base - 1 + i));
                }
                sum += w[i] * image[sx][sy];
            }
            sampled[x][y] = sum / wsum;
        }
    }
    resample_image(sampled, target_width, target_height, res);
}

matrix gaussian;

void prepare_gaussian_2d()
{
    const double sigma = 2;
    const int n = 2 * gradient_radius;
    const double center = gradient_radius - 0.5;
    double sum = 0;
    gaussian.assign(n, vector(n, 0.0));
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) {
            double dist = (i - center) * (i - center) + (j - center) * (j - center);
            double cost = exp(-dist / (2 * sigma * sigma));
            sum += cost;
            gaussian[i][j] = cost;
        }
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            gaussian[i][j] /= sum;
}

hashkey get_hashkey(const matrix &patch)
{
    const int n = 2 * radius;
    assert(patch.size() == n && patch[0].size() == n);

    assert(gradient_radius <= radius);
    const int gradient_left = radius - gradient_radius;
    const int gradient_right = n - gradient_left;

    // Eigenvalues and eigenvectors of 2x2 matrices
    // http://www.math.harvard.edu/archive/21b_fall_04/exhibits/2dmatrices/
    double a = 0, b = 0, d = 0;
    for (int i = gradient_left; i < gradient_right; i++)
        for (int j = gradient_left; j < gradient_right; j++) {
            double gx, gy;
            if (i == 0)
                gx = patch[i + 1][j] - patch[i][j];
            else if (i == n - 1)
                gx = patch[i][j] - patch[i - 1][j];
            else if (i == 1 || i == n - 2)
                gx = (patch[i + 1][j] - patch[i - 1][j]) / 2.0;
            else
                gx = (-patch[i + 2][j] + 8 * patch[i + 1][j] - 8 * patch[i - 1][j] + patch[i - 2][j]) / 12.0;
            if (j == 0)
                gy = patch[i][j + 1] - patch[i][j];
            else if (j == n - 1)
                gy = patch[i][j] - patch[i][j - 1];
            else if (j == 1 || j == n - 2)
                gy = (patch[i][j + 1] - patch[i][j - 1]) / 2.0;
            else
                gy = (-patch[i][j + 2] + 8 * patch[i][j + 1] - 8 * patch[i][j - 1] + patch[i][j - 2]) / 12.0;
            double gw = gaussian[i - gradient_left][j - gradient_left];
            a += gx * gx * gw;
            b += gx * gy * gw;
            d += gy * gy * gw;
        }
    // b == c
    const double T = a + d, D = a * d - b * b;
    const double delta = sqrt(fmax(T * T / 4 - D, 0));
    const double L1 = T / 2 + delta, L2 = T / 2 - delta;
    double V1x, V1y;
    if (fabs(b) > eps) {
        V1x = b;
        V1y = L1 - a;
    } else {
        V1x = 1;
        V1y = 0;
    }

    const double sqrtL1 = sqrt(L1), sqrtL2 = sqrt(L2);

    const double theta = fmod(atan2(V1y, V1x) + M_PI, M_PI);
    const double lambda = sqrtL1;
    const double mu = sqrtL1 + sqrtL2 < eps ? 0 : (sqrtL1 - sqrtL2) / (sqrtL1 + sqrtL2);

    const int angle = (int)floor(theta / (M_PI / quant_angle) + eps);
    assert(0 <= angle && angle < quant_angle);
    const int strength = std::lower_bound(min_strength, min_strength + quant_strength - 1, lambda) - min_strength;
    const int coherence = std::lower_bound(min_coherence, min_coherence + quant_coherence - 1, mu) - min_coherence;

    assert(0 <= strength && strength < quant_strength);
    assert(0 <= coherence && coherence < quant_coherence);
    hashkey ret(angle, strength, coherence);
    return ret;
}

double learning_rate = 0.001;
int batch_size = 64;
bool squared_error = false;
bool keep = false;

struct linear_optimizer {
    vector weights;

    vector accumulated_grad;

    vector grad;
    int instances;

    void reset() {
        weights.assign(feature_count, 0.0);

        // initialized with bilinear kernel.
        for (int x = 0; x < lut_size; x++) {
            for (int y = 0; y < lut_size; y++) {
                double fx = (double)x / (lut_size - 1);
                double fy = (double)y / (lut_size - 1);
                for (int i = 0; i < 2; i++) {
                    for (int j = 0; j < 2; j++) {
                        int var = feature_mapping[x][y][radius - 1 + i][radius - 1 + j];
                        if (var < 0)
                            continue;
                        double factor = i == 0 ? 1 - fx : fx;
                        factor *= j == 0 ? 1 - fy : fy;
                        weights[var] = factor;
                    }
                }
            }
        }

        if (keep)
            accumulated_grad.resize(feature_count, 0.0);
        else
            accumulated_grad.assign(feature_count, 0.0);

        start_batch();
    }

    void read(FILE *f) {
        reset();

        if ((int)fread(&*weights.begin(), sizeof(*weights.begin()), weights.size(), f) != feature_count) {
            DIE("failed to read model");
        }

        for (int i = 0; i < feature_count; i++) {
            double f = weights[i];
            if (std::isnan(f) || f < -1.0 || f > 2.0) {
                DIE("bad model (%.10f)\n", f);
            }
        }
    }

    /*void read(FILE *f) {
        reset();

        for (int x = 0; x < lut_size; x++) {
            for (int y = 0; y < lut_size; y++) {
                for (int i = 0; i < radius * 2; i++) {
                    for (int j = 0; j < radius * 2; j++) {
                        double val = 0.0;
                        if (fread(&val, sizeof(val), 1, f) != 1) {
                            DIE("failed to read model");
                        }
                        int var = feature_mapping[x][y][i][j];
                        if (var >= 0)
                            weights[var] = val;
                    }
                }
            }
        }

        for (int i = 0; i < feature_count; i++) {
            double f = weights[i];
            if (std::isnan(f) || f < -1.0 || f > 2.0) {
                DIE("bad model (%.10f)\n", f);
            }
        }
    }*/

    void write(FILE *f) {
        if ((int)fwrite(&*weights.begin(), sizeof(*weights.begin()), weights.size(), f) != feature_count) {
            DIE("failed to write model");
        }
    }

    /*
    void write(FILE *f) {
        int nradius = 4;
        for (int x = 0; x < lut_size; x++) {
            for (int y = 0; y < lut_size; y++) {
                for (int i = 0; i < nradius * 2; i++) {
                    for (int j = 0; j < nradius * 2; j++) {
                        double val = 0.0;
                        int ni = i - nradius + radius;
                        int nj = j - nradius + radius;
                        if (ni >= 0 && ni < radius * 2 && nj >= 0 && nj < radius * 2) {
                            int var = feature_mapping[x][y][i][j];
                            if (var >= 0) {
                                val = weights[feature_mapping[x][y][ni][nj]];
                            } else if (var == VAR_CONSTANT_1) {
                                val = 1.0;
                            }
                        }
                        if (fwrite(&val, sizeof(val), 1, f) != 1) {
                            DIE("failed to write model");
                        }
                    }
                }
            }
        }
    }*/

    vector get_model_weights() {
        vector ret;

        for (int x = 0; x < lut_size; x++) {
            for (int y = 0; y < lut_size; y++) {
                for (int i = 0; i < radius; i++) {
                    for (int j = 0; j < radius * 2; j++) {
                        int var = feature_mapping[x][y][i][j];
                        if (var == VAR_CONSTANT_1) {
                            ret.push_back(1.0);
                        } else if (var == VAR_CONSTANT_0) {
                            ret.push_back(0.0);
                        } else {
                            assert(0 <= var && var < (int)weights.size());
                            ret.push_back(weights[var]);
                        }
                    }
                }
            }
        }
        return ret;
    }

    void start_batch() {
        grad.assign(feature_count, 0.0);
        instances = 0;
    }

    void end_batch() {
        assert(instances >= batch_size);

        for (int i = 0; i < feature_count; i++) {
            double g = grad[i] / instances;
            double &h = accumulated_grad[i];

            h += g * g;
#if AR == 1
            weights[i] = std::max(0.0, weights[i] - learning_rate * g / (sqrt(h) + 1e-8));
#else
            weights[i] -= learning_rate * g / (sqrt(h) + 1e-8);
#endif
        }
    }

    double update(const std::vector<std::pair<int,double>> &A, double B) {
        instances ++;

        double res = 0;
        for (const auto &p : A)
            res += weights[p.first] * p.second;

        double err = res - B;

        if (squared_error) {
            for (const auto &p : A) {
                grad[p.first] += p.second * err;
            }
        } else {
            for (const auto &p : A) {
                grad[p.first] += p.second * (err > 0 ? 1 : -1);
            }
        }

        if (instances >= batch_size) {
            end_batch();
            start_batch();
        }

        return err;
    }

};

struct ravu_model {
    linear_optimizer models[quant_angle][quant_strength][quant_coherence];

    void reset() {
        for (int i = 0; i < quant_angle; i++)
            for (int j = 0; j < quant_strength; j++)
                for (int k = 0; k < quant_coherence; k++)
                    models[i][j][k].reset();
    }

    void read(const std::string &file_name) {
        FILE* f = fopen(file_name.c_str(), "rb");

        if (!f) {
            reset();
            write(file_name);
            return;
        }

        for (int i = 0; i < quant_angle; i++)
            for (int j = 0; j < quant_strength; j++)
                for (int k = 0; k < quant_coherence; k++)
                    models[i][j][k].read(f);

        fclose(f);
    }

    void write(const std::string &file_name) {
        FILE* f = fopen(file_name.c_str(), "wb");

        if (!f) {
            DIE("unable to write to '%s'\n", file_name.c_str());
            return;
        }

        for (int i = 0; i < quant_angle; i++)
            for (int j = 0; j < quant_strength; j++)
                for (int k = 0; k < quant_coherence; k++)
                    models[i][j][k].write(f);

        fclose(f);
    }

    void visualize(const std::string &bmp_file) {
        const int block_size = radius * 2 * (lut_size - 1) + 1;
        const int border_size = 1;
        const int width = quant_angle * (block_size + border_size * 2);
        const int height = quant_coherence * quant_strength * (block_size + border_size * 2);

        std::vector<std::vector<double>> img_sum(width, std::vector<double>(height, 0.0));
        std::vector<std::vector<int>> img_cnt(width, std::vector<int>(height, 0));

        for (int angle = 0; angle < quant_angle; angle++) {
            for (int strength = 0; strength < quant_strength; strength++) {
                for (int coherence = 0; coherence < quant_coherence; coherence++) {
                    for (int x = 0; x < radius * 2; x++) {
                        for (int y = 0; y < radius * 2; y++) {
                            int nx = angle * (block_size + border_size * 2) + x * (lut_size - 1) + border_size;
                            int ny = (strength + quant_strength * coherence) * (block_size + border_size * 2) + y * (lut_size - 1) + border_size;
                            for (int lut_x = 0; lut_x < lut_size; lut_x++) {
                                for (int lut_y = 0; lut_y < lut_size; lut_y++) {
                                    int var = feature_mapping[lut_size - 1 - lut_x][lut_size - 1 - lut_y][x][y];
                                    double w = 0;
                                    if (var == VAR_CONSTANT_1) {
                                        w = 1;
                                    } else if (var >= 0) {
                                        w = models[angle][strength][coherence].weights[var];
                                    }
                                    img_sum[nx + lut_x][ny + lut_y] += w;
                                    img_cnt[nx + lut_x][ny + lut_y] ++;
                                }
                            }
                        }
                    }
                }
            }
        }

        unsigned long file_size = bmp_size(width, height);

        std::vector<uint8_t> bmp_buffer(file_size, 0);
        bmp_init(&bmp_buffer[0], width, height);

        for (int x = 0; x < width; x++) {
            for (int y = 0; y < height; y++) {
                double w = img_cnt[x][y] == 0 ? 0.0 : img_sum[x][y] / img_cnt[x][y];
                w *= 4;
                w /= 1 + fabs(w);
                unsigned long color;
                if (w < 0) {
                    w = -w;
                    color = bmp_encode(1, 1-w, 1-w); // red
                } else {
                    color = bmp_encode(1-w, 1-w, 1); // blue
                }
                bmp_set(&bmp_buffer[0], x, y, color);
            }
        }

        FILE *f = fopen(bmp_file.c_str(), "wb");
        if (!f) {
            DIE("failed to open '%s'\n", bmp_file.c_str());
            return;
        }
        fwrite(&bmp_buffer[0], 1, file_size, f);
        fclose(f);
    }

    void upscale(const matrix &lowres, int width_hi, int height_hi, matrix *hires) {
        int width_lo = lowres[0].size(), height_lo = lowres.size();

        hires->assign(height_hi, vector(width_hi));

        std::vector<std::vector<std::vector<point_in_pixel>>> points_by_location(height_lo);
        for (int i = 0; i < height_lo; i++)
            points_by_location[i].resize(width_lo);

        for (int x = 0; x < height_hi; x++) {
            for (int y = 0; y < width_hi; y++) {
                double mapped_x = (double)(x + 0.5) / height_hi * height_lo - 0.5;
                double mapped_y = (double)(y + 0.5) / width_hi * width_lo - 0.5;
                int base_x = std::max(0, std::min(height_lo - 1, (int)floor(mapped_x)));
                int base_y = std::max(0, std::min( width_lo - 1, (int)floor(mapped_y)));
                points_by_location[base_x][base_y].push_back(
                        point_in_pixel(mapped_x - base_x, mapped_y - base_y, x, y));
            }
        }

        matrix patch(radius * 2, vector(radius * 2));

        for (int x = 0; x < height_lo; x++) {
            for (int y = 0; y < width_lo; y++) {

                for (int i = 0; i < radius * 2; i ++) {
                    for (int j = 0; j < radius * 2; j++) {
                        int nx = std::max(0, std::min(height_lo - 1, x - (radius - 1) + i));
                        int ny = std::max(0, std::min( width_lo - 1, y - (radius - 1) + j));
                        patch[i][j] = lowres[nx][ny];
                    }
                }

                hashkey key = get_hashkey(patch);
                const linear_optimizer &model = models[key.angle][key.strength][key.coherence];

                for (const auto &point : points_by_location[x][y]) {
                    int lut_x = std::min(lut_size - 2, std::max(0, (int)floor(point.x * (lut_size - 1))));
                    int lut_y = std::min(lut_size - 2, std::max(0, (int)floor(point.y * (lut_size - 1))));
                    double sub_x = point.x * (lut_size - 1) - lut_x;
                    double sub_y = point.y * (lut_size - 1) - lut_y;

                    double res = 0;

                    for (int i = 0; i < 2; i++) {
                        for (int j = 0; j < 2; j++) {
                            double factor = i == 0 ? 1 - sub_x : sub_x;
                            factor *= j == 0 ? 1 - sub_y : sub_y;
                            for (int u = 0; u < radius * 2; u++) {
                                for (int v = 0; v < radius * 2; v++) {
                                    int var = feature_mapping[lut_x + i][lut_y + j][u][v];
                                    if (var == VAR_CONSTANT_1) {
                                        res += patch[u][v] * factor;
                                    } else if (var >= 0) {
                                        res += patch[u][v] * factor * model.weights[var];
                                    }
                                }
                            }
                        }
                    }

                    (*hires)[point.origin[0]][point.origin[1]] = res;
                }
            }
        }
    }

    void train(const matrix &hires, const matrix &lowres) {
        int width_hi = hires[0].size(), height_hi = hires.size();
        int width_lo = lowres[0].size(), height_lo = lowres.size();

        std::vector<std::pair<int,int>> location_by_key[quant_angle][quant_strength][quant_coherence];

        matrix patch(radius * 2, vector(radius * 2));

        for (int x = 0; x + radius * 2 <= height_lo; x++) {
            for (int y = 0; y + radius * 2 <= width_lo; y++) {
                for (int i = 0; i < radius * 2; i ++) {
                    for (int j = 0; j < radius * 2; j++) {
                        patch[i][j] = lowres[x + i][y + j];
                    }
                }
                hashkey key = get_hashkey(patch);
                location_by_key[key.angle][key.strength][key.coherence].push_back({x+radius-1, y+radius-1});
            }
        }

        std::vector<std::vector<std::vector<point_in_pixel>>> points_by_location(height_lo);
        for (int i = 0; i < height_lo; i++)
            points_by_location[i].resize(width_lo);

        for (int x = 0; x < height_hi; x++) {
            for (int y = 0; y < width_hi; y++) {
                double mapped_x = (double)(x + 0.5) / height_hi * height_lo - 0.5;
                double mapped_y = (double)(y + 0.5) / width_hi * width_lo - 0.5;
                int base_x = (int)floor(mapped_x);
                int base_y = (int)floor(mapped_y);
                if (base_x < 0 || base_y < 0)
                    continue;
                points_by_location[base_x][base_y].push_back(
                        point_in_pixel(mapped_x - base_x, mapped_y - base_y, hires[x][y]));
            }
        }

        for (int angle = 0; angle < quant_angle; angle++) {
            for (int strength = 0; strength < quant_strength; strength++) {
                for (int coherence = 0; coherence < quant_coherence; coherence++) {
                    linear_optimizer &solver = models[angle][strength][coherence];
                    for (const auto &loc : location_by_key[angle][strength][coherence]) {
                        int x = loc.first, y = loc.second;
                        for (int i = 0; i < radius * 2; i++) {
                            for (int j = 0; j < radius * 2; j++) {
                                patch[i][j] = lowres[x - (radius - 1) + i][y - (radius - 1) + j];
                            }
                        }
                        for (const auto &point : points_by_location[x][y]) {
                            int lut_x = std::min(lut_size - 2, std::max(0, (int)floor(point.x * (lut_size - 1))));
                            int lut_y = std::min(lut_size - 2, std::max(0, (int)floor(point.y * (lut_size - 1))));
                            double sub_x = point.x * (lut_size - 1) - lut_x;
                            double sub_y = point.y * (lut_size - 1) - lut_y;

                            std::vector<std::pair<int,double>> A;
                            double B = point.val;

                            for (int i = 0; i < 2; i++) {
                                for (int j = 0; j < 2; j++) {
                                    double factor = i == 0 ? 1 - sub_x : sub_x;
                                    factor *= j == 0 ? 1 - sub_y : sub_y;
                                    for (int u = 0; u < radius * 2; u++) {
                                        for (int v = 0; v < radius * 2; v++) {
                                            int var = feature_mapping[lut_x + i][lut_y + j][u][v];
                                            if (var == VAR_CONSTANT_1) {
                                                B -= patch[u][v] * factor;
                                            } else if (var >= 0) {
                                                A.push_back({var, patch[u][v] * factor});
                                            }
                                        }
                                    }
                                }
                            }
                            solver.update(A, B);
                        }
                    }
                }
            }
        }
    }

    void train(const matrix &hires, double ratio) {
        int width = hires[0].size(), height = hires.size();
        matrix lowres;
        resample_image(hires, width / ratio, height / ratio, &lowres);
        train(hires, lowres);
    }

    std::vector<std::vector<std::vector<vector>>> get_model_weights() {
        std::vector<std::vector<std::vector<vector>>> ret;
        ret.resize(quant_angle);
        for (int i = 0; i < quant_angle; i++) {
            ret[i].resize(quant_strength);
            for (int j = 0; j < quant_strength; j++) {
                ret[i][j].resize(quant_coherence);
                for (int k = 0; k < quant_coherence; k++) {
                    ret[i][j][k] = models[i][j][k].get_model_weights();
                }
            }
        }
        return ret;
    }

    void write_model(FILE *file) {
#if AR == 0
        fprintf(file, "radius = %d\n", radius);
        fprintf(file, "lut_size = %d\n", lut_size);
        fprintf(file, "gradient_radius = %d\n", gradient_radius);
        fprintf(file, "quant_angle = %d\n", quant_angle);
        fprintf(file, "quant_strength = %d\n", quant_strength);
        fprintf(file, "quant_coherence = %d\n", quant_coherence);
        fprintf(file, "min_strength = %s\n", to_string(vector(min_strength, min_strength + quant_strength - 1)).c_str());
        fprintf(file, "min_coherence = %s\n", to_string(vector(min_coherence, min_coherence + quant_coherence - 1)).c_str());
        fprintf(file, "gaussian = %s\n", to_string(gaussian).c_str());
        fprintf(file, "model_weights = %s\n", to_string(get_model_weights()).c_str());
#else
        fprintf(file, "model_weights_ar = %s\n", to_string(get_model_weights()).c_str());
#endif
    }

};

ravu_model model;

double evaluate(const std::vector<std::string> file_names, bool detail)
{
    if (file_names.empty())
        return 0.0;
    const double ratio = sqrt(5.0);
    double sum_psnr = 0;
    for (const auto& f : file_names) {
        matrix hires = read_image(f);

        int width = hires[0].size(), height = hires.size();
        matrix lowres, bicubic_upscaled, ravu_upscaled;

        resample_image(hires, width / ratio, height / ratio, &lowres);
        resample_image(lowres, width, height, &bicubic_upscaled);
        model.upscale(lowres, width, height, &ravu_upscaled);

        double bicubic_psnr = psnr(hires, bicubic_upscaled);
        double ravu_psnr = psnr(hires, ravu_upscaled);

        if (detail)
            PRINT("bicubic: %.6lf, ravu: %.6lf\n", bicubic_psnr, ravu_psnr);

        sum_psnr += ravu_psnr - bicubic_psnr;
    }

    double avg_psnr = sum_psnr / file_names.size();

    PRINT("avg psnr improve: %.6lf\n", avg_psnr);

    return avg_psnr;
}

int main(int argc, char *argv[])
{
    srand(time(0));
    prepare_feature_mapping();
    prepare_gaussian_2d();

    std::string model_filename = "model.bin";

    int argv_ptr = 1;

    std::vector<std::string> train, eval;
    bool write = false, upscale = false, vis = false;

    std::string upscale_from, upscale_to;
    int upscale_width = 0, upscale_height = 0;

    while (argv_ptr < argc) {
        bool need1 = argv_ptr + 1 < argc;
        bool need4 = argv_ptr + 4 < argc;
        const char *cur = argv[argv_ptr];
        const char *next = argv[argv_ptr + 1];
        if (need1 && strcmp(cur, "--model") == 0) {
            model_filename = next;
            argv_ptr += 2;
        } else if (need1 && strcmp(cur, "--lr") == 0) {
            sscanf(next, "%lf", &learning_rate);
            argv_ptr += 2;
        } else if (need1 && strcmp(cur, "--batch-size") == 0) {
            sscanf(next, "%d", &batch_size);
            argv_ptr += 2;
        } else if (need1 && strcmp(cur, "--train") == 0) {
            for (++argv_ptr; argv_ptr < argc && argv[argv_ptr][0] != '-'; ) {
                train.push_back(argv[argv_ptr++]);
            }
        } else if (need1 && strcmp(cur, "--eval") == 0) {
            for (++argv_ptr; argv_ptr < argc && argv[argv_ptr][0] != '-'; ) {
                eval.push_back(argv[argv_ptr++]);
            }
        } else if (strcmp(cur, "--write") == 0) {
            write = true;
            argv_ptr ++;
        } else if (strcmp(cur, "--sqerr") == 0) {
            squared_error = true;
            argv_ptr++;
        } else if (strcmp(cur, "--keep") == 0) {
            keep = true;
            argv_ptr++;
        } else if (need4 && strcmp(cur, "--upscale") == 0) {
            upscale = true;
            upscale_from = next;
            upscale_to = argv[argv_ptr + 2];
            upscale_width = atoi(argv[argv_ptr + 3]);
            upscale_height = atoi(argv[argv_ptr + 4]);
            argv_ptr += 5;
        } else if (strcmp(cur, "--vis") == 0) {
            vis = true;
            argv_ptr ++;
        } else {
            break;
        }
    }

    if (argc == 1 || argv_ptr < argc) {
        PRINT("Usage: %s [--model model.bin] [--lr lr] [--batch-size bs] [--sqerr] [--keep]"
              "[--train files..] [--eval files..] [--write] [--vis]"
              "[--upscale low high w h]\n", argv[0]);
        return -1;
    }

    model.read(model_filename);

    if (write) {
        model.write_model(stdout);
        return 0;
    }

    if (upscale) {
        matrix image = read_image(upscale_from);
        matrix upscaled;
        model.upscale(image, upscale_width, upscale_height, &upscaled);
        write_image(upscale_to, upscaled);
        return 0;
    }

    if (vis) {
        std::string f = remove_suffix(model_filename, ".bin");
        f += ".vis";
        f += ".bmp";
        model.visualize(f);
        return 0;
    }

    PRINT("model=%s, learning_rate=%.9lf, batch_size=%d, method=%s, keep=%d, ar=%d\n",
          model_filename.c_str(), learning_rate, batch_size, squared_error ? "squared" : "absolute", keep, AR);

    if (train.size() > 0) {
        double max_psnr = 0.0;
        std::string max_fn = "";
        if (eval.size() > 0) {
            double psnr = evaluate(eval, false);
            if (psnr > max_psnr) {
                PRINT("initial psnr is %.6lf\n", psnr)
                max_psnr = psnr;
                max_fn = remove_suffix(model_filename, ".bin") + "-input.bin";
                model.write(max_fn);
            }
        }
        for (int iteration = 1; ; iteration++) {
            random_shuffle(train.begin(), train.end());
            for (int i = 0; i < (int)train.size(); i++) {
                matrix image = read_image(train[i]);
                double ratio = exp2((double)rand() / RAND_MAX) * 1.5;
                model.train(image, ratio);
                if ((i + 1) % 10 == 0) {
                    model.write(model_filename);
                    model.write(model_filename + ".bak");
                }
            }
            if (eval.size() > 0) {
                double psnr = evaluate(eval, false);
                if (psnr > max_psnr) {
                    PRINT("new best found: %.6lf\n", psnr)
                    max_psnr = psnr;
                    std::string f = remove_suffix(model_filename, ".bin");
                    char buf[100];
                    snprintf(buf, sizeof(buf), "%.6lf", psnr);
                    f += "-";
                    f += buf;
                    f += ".bin";
                    model.write(f);
                    max_fn = f;
                    iteration = 0;
                } else if (iteration >= 10) {
                    learning_rate *= 0.9;
                    PRINT("model reverted to %s, set new learning rate to %.6lf\n", max_fn.c_str(), learning_rate);
                    model.read(max_fn);
                    iteration = 0;
                }
            }
        }
    } else if (eval.size() > 0) {
        evaluate(eval, true);
    }
}
