//
// Frontplatte 200 x 100 mm für DFRobot Embedded Thermal Printer V2.0
// MODIFIZIERTE VERSION:
// - Seitenwände bleiben keilförmig
// - die obere schräge Ebene ist NICHT mehr vollflächig
// - stattdessen nur noch zwei schräge Randstege à 10 mm
// - mittlerer Bereich oben bleibt offen
// - Steigung standardmäßig als 15 % statt 15°
// - mittiger Printer-Cutout
// - front-face-down druckbar
//

part = "plate";   // "plate" | "preview"

// =========================
// HAUPTPARAMETER
// =========================

panel_w = 200.0;
panel_h = 100.0;

plate_t = 2.2;
outer_corner_r = 2.0;

rim_inset = 1.0;
rim_t = 1.2;
rim_depth = 20.0;

// Obere Schräge:
// Standard = 15 % Steigung über rim_depth
use_slope_percent = true;
top_mount_slope_percent = 400.0;

// Nur falls du doch Winkel statt Prozent willst:
top_mount_angle_from_front = 15.0;

top_wall_t = rim_t;

// neue schräge Randstege
top_rail_w = 10.0;   // jeweils 1 cm breit

// DFRobot Printer Cutout
printer_install_w = 77.2;
printer_install_h = 53.2;
printer_open_extra_w = 0.8;
printer_open_extra_h = 0.8;

printer_open_w = printer_install_w + printer_open_extra_w;
printer_open_h = printer_install_h + printer_open_extra_h;
printer_cutout_r = 1.5;

// Dummy preview
dummy_body_w = 77.2;
dummy_body_h = 53.2;
dummy_body_d = 42.0;

dummy_bezel_w = 82.1;
dummy_bezel_h = 58.1;
dummy_bezel_t = 2.0;

// =========================
// ABGELEITETE WERTE
// =========================

top_mount_rise = use_slope_percent
    ? rim_depth * top_mount_slope_percent / 100
    : rim_depth * tan(90 - top_mount_angle_from_front);

y_bottom    = -panel_h/2 + rim_inset;
y_top_inner =  panel_h/2 - rim_inset;

// obere schräge Stege
y_front_center = y_top_inner - top_wall_t/2;
y_back_center  = y_front_center + top_mount_rise;

y_front_low  = y_front_center - top_wall_t/2;
y_front_high = y_front_center + top_wall_t/2;
y_back_low   = y_back_center  - top_wall_t/2;
y_back_high  = y_back_center  + top_wall_t/2;

// X-Positionen der Seitenwände
x_left_outer  = -panel_w/2 + rim_inset;
x_right_outer =  panel_w/2 - rim_inset;

// X-Positionen der oberen Randstege
x_left_rail_center  = x_left_outer + top_rail_w/2;
x_right_rail_center = x_right_outer - top_rail_w/2;

// =========================
// 2D-HELFER
// =========================

module rounded_rect_2d(w, h, r) {
    rr = min(r, min(w, h) / 2);
    if (rr <= 0) {
        square([w, h], center = true);
    } else {
        hull() {
            for (x = [-w/2 + rr, w/2 - rr])
                for (y = [-h/2 + rr, h/2 - rr])
                    translate([x, y]) circle(r = rr, $fn = 32);
        }
    }
}

module printer_cutout_2d() {
    rounded_rect_2d(printer_open_w, printer_open_h, printer_cutout_r);
}

// =========================
// FRONTPLATTE
// =========================

module front_plate() {
    linear_extrude(height = plate_t, convexity = 10)
        difference() {
            rounded_rect_2d(panel_w, panel_h, outer_corner_r);
            printer_cutout_2d();
        }
}

// =========================
// SEITENWÄNDE ALS KEILE
// =========================

module side_wall_wedge(x_outer) {
    eps = 0.02;

    front_h = y_top_inner - y_bottom;
    back_h  = y_back_high - y_bottom;

    hull() {
        // vorderer Rechteck-Streifen
        translate([
            x_outer + rim_t/2,
            (y_bottom + y_top_inner)/2,
            plate_t + eps/2
        ])
            cube([rim_t, front_h, eps], center = true);

        // hinterer höherer Rechteck-Streifen
        translate([
            x_outer + rim_t/2,
            (y_bottom + y_back_high)/2,
            plate_t + rim_depth - eps/2
        ])
            cube([rim_t, back_h, eps], center = true);
    }
}

module left_wall() {
    side_wall_wedge(x_left_outer);
}

module right_wall() {
    side_wall_wedge(x_right_outer - rim_t);
}

// =========================
// BODENWAND
// =========================

module bottom_wall() {
    translate([
        -panel_w/2 + rim_inset + rim_t,
        y_bottom,
        plate_t
    ])
        cube([
            panel_w - 2*(rim_inset + rim_t),
            rim_t,
            rim_depth
        ], center = false);
}

// =========================
// OBERE SCHRÄGE RANDSTEGE
// =========================
//
// Keine Vollfläche mehr.
// Nur noch links und rechts je ein 10-mm-Steg.
//

module top_sloped_rail(x_center, rail_w) {
    eps = 0.02;

    hull() {
        // vorderer Streifen direkt an der Rückseite der Frontplatte
        translate([x_center, y_front_center, plate_t + eps/2])
            cube([rail_w, top_wall_t, eps], center = true);

        // hinterer Streifen weiter hinten und höher
        translate([x_center, y_back_center, plate_t + rim_depth - eps/2])
            cube([rail_w, top_wall_t, eps], center = true);
    }
}

module left_top_rail() {
    top_sloped_rail(x_left_rail_center, top_rail_w);
}

module right_top_rail() {
    top_sloped_rail(x_right_rail_center, top_rail_w);
}

// =========================
// PRINTER-CUTOUT 3D
// =========================

module printer_cutout_3d() {
    translate([0, 0, -0.1])
        linear_extrude(height = plate_t + rim_depth + 0.2, convexity = 10)
            printer_cutout_2d();
}

// =========================
// GESAMTTEIL
// =========================

module plate() {
    difference() {
        union() {
            front_plate();
            left_wall();
            right_wall();
            bottom_wall();
            left_top_rail();
            right_top_rail();
        }

        printer_cutout_3d();
    }
}

// =========================
// PREVIEW
// =========================

module dummy_printer() {
    color([0.15, 0.15, 0.15, 0.55])
        union() {
            // Druckerkörper hinter der Platte
            translate([0, 0, plate_t])
                linear_extrude(height = dummy_body_d, convexity = 10)
                    rounded_rect_2d(dummy_body_w, dummy_body_h, 1.0);

            // Frontüberstand vor der Platte
            translate([0, 0, -dummy_bezel_t])
                linear_extrude(height = dummy_bezel_t, convexity = 10)
                    rounded_rect_2d(dummy_bezel_w, dummy_bezel_h, 1.5);
        }
}

module preview() {
    plate();
    dummy_printer();
}

// =========================
// OUTPUT
// =========================

if (part == "preview") {
    preview();
} else {
    plate();
}