//
// ReSpeaker XVF3800 + XIAO ESP32S3
// OVAL TOP CAP WITH HIDDEN CABLE CHANNEL
//
// Konzept:
// - nur Oberteil / Haube
// - innen: runder Mesh-Bereich über dem ReSpeaker
// - außen: geschlossener ovaler Kabelkanal
// - außen keine Öffnungen
// - unten komplett offen: Kabel werden unten aus dem Oval nach unten abgeführt
// - innere Kreiswand bekommt Reliefs für USB-C / Audio / Speaker / Buttons
// - keine Schrauben
//
// Druck:
// - mit Top-Seite / Mesh nach unten aufs Bett
// - kein Support
//
// Export:
//   part = "cap";
//   part = "preview";
//
// FIRST-PASS:
// - board_d ist bewusst parametrierbar
// - erst Test-Fit drucken
//

part = "cap";   // "cap" | "preview"

// =========================
// HAUPTPARAMETER
// =========================

// ReSpeaker PCB - First pass
board_d = 99.0;
board_t = 4.0;

// Passung am PCB
fit_clear = 0.45;   // radialer Abstand PCB -> Innenwand
grip_depth = 0.45;  // leichte Klemmung über Grip-Ribs
grip_w = 8.0;
grip_h = 2.8;

// Wandstärken - bewusst schlanker als vorher
device_wall_t = 1.60;   // innere Kreiswand um das Gerät
outer_wall_t  = 1.80;   // äußere Ovalwand
face_t        = 1.00;   // Top-Dicke
top_gap       = 1.20;   // Abstand Mesh-Unterseite -> PCB-Oberseite
comp_clear    = 13.00;  // Tiefe nach unten für XIAO, USB-C-Stecker, etc.

// Ovaler Außenkanal
side_channel_w = 8.00;   // seitliche Kanalbreite links/rechts
tb_extra       = 16.00;  // zusätzliche Ausdehnung oben/unten

// Gesamthöhe
cap_h = face_t + top_gap + board_t + comp_clear;

// =========================
// MESH
// =========================

mesh_pitch        = 5.60;
mesh_bar_w        = 1.20;
mesh_center_hub_r = 5.50;
mesh_angle_a      = 45;
mesh_angle_b      = -45;
mesh_rim_w        = 2.00;

// =========================
// BOARD-SITZ / HALTUNG
// =========================

// Board liegt auf 4 Segmentauflagen
shelf_inset  = 1.10;
shelf_t      = 0.90;
shelf_arc    = 24;
shelf_angles = [45, 135, 225, 315];

// zusätzliche Grip-Ribs gegen Wackeln
grip_z0      = face_t + top_gap + 0.70;
grip_angles  = [20, 160, 200, 340];

// =========================
// EDGE-RELIEFS IN DER INNEREN KREISWAND
// =========================

// Winkel:
// 0° = rechts
// 90° = oben
// 180° = links
// 270° = unten

xiao_usb_angle = 90;
xmos_usb_angle = 270;
audio_angle    = 240;
speaker_angle  = 332;
mute_angle     = 0;
reset_angle    = 180;

// tangentiale Breiten der Reliefs
xiao_usb_w = 15.0;
xmos_usb_w = 15.0;
audio_w    = 12.0;
speaker_w  = 14.0;
button_w   =  5.8;

// =========================
// ABGELEITETE MASSE
// =========================

pcb_top_z    = face_t + top_gap;
pcb_bottom_z = pcb_top_z + board_t;

device_inner_r = board_d/2 + fit_clear;
device_outer_r = device_inner_r + device_wall_t;

// Außenoval = "Kapsel" / Stadium
outer_capsule_r    = device_outer_r + side_channel_w;
outer_capsule_offy = tb_extra;

// Mesh nur im inneren Kreis
mesh_outer_r = device_outer_r - 1.20;
mesh_field_r = mesh_outer_r - mesh_rim_w;

// Relief-Zonen
port_z0    = pcb_top_z + 0.40;
button_z0  = pcb_top_z + 0.80;
button_z1  = button_z0 + 4.50;

inner_relief_depth = device_wall_t + 6.0;

// =========================
// 2D-HELFER
// =========================

module capsule2d(r, offy) {
    hull() {
        translate([0,  offy]) circle(r = r, $fn = 180);
        translate([0, -offy]) circle(r = r, $fn = 180);
    }
}

module stripe_field_2d(r, pitch, bar_w, angle_deg) {
    intersection() {
        circle(r = r, $fn = 220);
        union() {
            for (y = [-2*r : pitch : 2*r]) {
                translate([0, y])
                    rotate(angle_deg)
                        square([4*r, bar_w], center = true);
            }
        }
    }
}

module top_face_2d() {
    union() {
        // alles außerhalb des Mesh-Kreises ist geschlossen
        difference() {
            capsule2d(outer_capsule_r, outer_capsule_offy);
            circle(r = mesh_outer_r, $fn = 220);
        }

        // kleiner Innenring am Mesh-Rand für saubere Anbindung
        difference() {
            circle(r = mesh_outer_r, $fn = 220);
            circle(r = mesh_field_r, $fn = 220);
        }

        // verbundenes Diamond-Mesh
        stripe_field_2d(mesh_field_r + 0.02, mesh_pitch, mesh_bar_w, mesh_angle_a);
        stripe_field_2d(mesh_field_r + 0.02, mesh_pitch, mesh_bar_w, mesh_angle_b);

        // stabiler Kern
        circle(r = mesh_center_hub_r, $fn = 100);
    }
}

// =========================
// 3D-HELFER
// =========================

module top_face() {
    linear_extrude(height = face_t, convexity = 10)
        top_face_2d();
}

module inner_device_wall() {
    linear_extrude(height = cap_h, convexity = 10)
        difference() {
            circle(r = device_outer_r, $fn = 240);
            circle(r = device_inner_r, $fn = 240);
        }
}

module outer_oval_wall() {
    linear_extrude(height = cap_h, convexity = 10)
        difference() {
            capsule2d(outer_capsule_r, outer_capsule_offy);
            capsule2d(outer_capsule_r - outer_wall_t, outer_capsule_offy);
        }
}

module ring_arc_segment(r_in, r_out, z0, h, center_angle, arc_angle) {
    rotate([0, 0, center_angle - arc_angle/2])
        translate([0, 0, z0])
            rotate_extrude(angle = arc_angle, convexity = 10, $fn = 96)
                translate([r_in, 0])
                    square([r_out - r_in, h], center = false);
}

module support_shelves() {
    for (a = shelf_angles) {
        ring_arc_segment(
            device_inner_r - shelf_inset,
            device_inner_r,
            pcb_bottom_z - shelf_t,
            shelf_t,
            a,
            shelf_arc
        );
    }
}

module grip_rib(angle_deg) {
    rotate([0, 0, angle_deg])
        translate([device_inner_r - grip_depth/2, 0, grip_z0 + grip_h/2])
            cube([grip_depth, grip_w, grip_h], center = true);
}

module grip_ribs() {
    for (a = grip_angles) {
        grip_rib(a);
    }
}

module wall_window(angle_deg, center_r, tangential_w, z0, z1, depth) {
    rotate([0, 0, angle_deg])
        translate([center_r, 0, (z0 + z1)/2])
            cube([depth, tangential_w, z1 - z0], center = true);
}

module inner_reliefs() {
    // beide USB-C Ports
    wall_window(
        xiao_usb_angle,
        device_outer_r - device_wall_t/2,
        xiao_usb_w,
        port_z0,
        cap_h + 0.10,
        inner_relief_depth
    );

    wall_window(
        xmos_usb_angle,
        device_outer_r - device_wall_t/2,
        xmos_usb_w,
        port_z0,
        cap_h + 0.10,
        inner_relief_depth
    );

    // AUX + Speaker JST - nur innere Reliefs, außen bleibt geschlossen
    wall_window(
        audio_angle,
        device_outer_r - device_wall_t/2,
        audio_w,
        port_z0,
        cap_h + 0.10,
        inner_relief_depth
    );

    wall_window(
        speaker_angle,
        device_outer_r - device_wall_t/2,
        speaker_w,
        port_z0,
        cap_h + 0.10,
        inner_relief_depth
    );

    // Mute / Reset
    wall_window(
        mute_angle,
        device_outer_r - device_wall_t/2,
        button_w,
        button_z0,
        button_z1,
        inner_relief_depth
    );

    wall_window(
        reset_angle,
        device_outer_r - device_wall_t/2,
        button_w,
        button_z0,
        button_z1,
        inner_relief_depth
    );
}

module cap() {
    difference() {
        union() {
            top_face();
            inner_device_wall();
            outer_oval_wall();
            support_shelves();
            grip_ribs();
        }

        // nur die innere Kreiswand wird geöffnet
        inner_reliefs();
    }
}

// =========================
// PREVIEW
// =========================

module dummy_board() {
    color([0.10, 0.60, 0.20, 0.35])
        translate([0, 0, pcb_top_z])
            cylinder(h = board_t, d = board_d, $fn = 240);
}

module preview() {
    cap();
    dummy_board();
}

// =========================
// OUTPUT
// =========================

if (part == "preview") {
    preview();
} else {
    cap();
}