/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "tiler_event_handler.hpp"

// #include "gstnvdsmeta.h"
#include <X11/Xlib.h>
#include <X11/Xutil.h>
#include <termios.h>
#include <unistd.h>
#include <iostream>
#include <gst/gst.h>
#include <gst/video/videooverlay.h>

static const uint DEFAULT_X_WINDOW_WIDTH = 1920;
static const uint DEFAULT_X_WINDOW_HEIGHT = 1080;
static const char APP_TITLE[] = "DeepStream";
static const char FONT_NAME[] = "Serif";

namespace deepstream {

    void set_x_window(uint64_t x_window, GstObject *element_);

    void set_x_window(uint64_t x_window, GstObject *element_)
    {
        gst_video_overlay_set_window_handle(GST_VIDEO_OVERLAY(element_),
                                            (gulong)x_window);
        gst_video_overlay_expose(GST_VIDEO_OVERLAY(element_));
    }

    // void NvDsTilerEventHandler::kb_event_handler_thread_func() {
    // // INvDsKeyboardInput *kb_input = kb_input_.try_get().value();
    // // FILE *fp = kb_input->get_fileptr();
    // while (!kb_event_thread_stop) {
    //     // Check for keyboard input
    //     if (!kb_input->is_data_available(50000)) {
    //     continue;
    //     }
    //     int c = fgetc(fp);
    //     std::cout << std::endl;
    //     gint source_id;
    //     guint rows, columns;
    //     g_object_get(G_OBJECT(tiler), "show-source", &source_id, "rows", &rows,
    //                 "columns", &columns, NULL);
    //     if (kb_selecting) {
    //     if (kb_row_selected == FALSE) {
    //         if (c >= '0' && c <= '9') {
    //         selected_row = c - '0';
    //         if (selected_row < rows) {
    //             printf("[TileEventHandler] Selected row %d\n", selected_row);
    //             kb_row_selected = TRUE;
    //         } else {
    //             printf("[TileEventHandler] Selected row %d out of bounds\n",
    //                 selected_row);
    //             kb_selecting = FALSE;
    //             kb_row_selected = FALSE;
    //         }
    //         }
    //     } else {
    //         if (c >= '0' && c <= '9') {
    //         selected_col = c - '0';
    //         if (selected_col < columns) {
    //             kb_selecting = FALSE;
    //             kb_row_selected = FALSE;
    //             source_id = columns * selected_row + selected_col;
    //             printf("[TileEventHandler] Selected column %d: source-id %d\n",
    //                 selected_col, source_id);
    //             set_active_source(source_id);
    //         } else {
    //             printf("[TileEventHandler] Selected column %d out of bounds\n",
    //                 selected_col);
    //             kb_selecting = FALSE;
    //             kb_row_selected = FALSE;
    //         }
    //         }
    //     }
    //     }
    //     if (c == 'z') {
    //     if (source_id == -1 && kb_selecting == FALSE) {
    //         GXF_LOG_INFO("[TileEventHandler] Selecting source\n");
    //         kb_selecting = TRUE;
    //     } else {
    //         set_active_source(-1);
    //         kb_selecting = FALSE;
    //         kb_row_selected = FALSE;
    //     }
    //     }
    // }
    // }
    // gxf_result_t NvDsTilerEventHandler::registerInterface(
    //     nvidia::gxf::Registrar *registrar) {
    // nvidia::gxf::Expected<void> result;
    // result &= registrar->parameter(
    //     tiler_sink_pad_probe_, "tiler-out", "Tiler Output",
    //     "Handle to a nvidia::deepstream::NvDsProbeConnector component.",
    //     gxf::Registrar::NoDefaultParameter(), GXF_PARAMETER_FLAGS_OPTIONAL);
    // result &= registrar->parameter(
    //     renderer_property_controller_, "renderer-prop-controller",
    //     "Renderer Property Controller",
    //     "Handle to a nvidia::deepstream::NvDsVideoRendererPropertyController "
    //     "component.",
    //     gxf::Registrar::NoDefaultParameter(), GXF_PARAMETER_FLAGS_OPTIONAL);
    // result &= registrar->parameter(
    //     osd_property_controller_, "nvdsosd-prop-controller",
    //     "NvDsOSD Property Controller",
    //     "Handle to a nvidia::deepstream::NvDsOSDPropertyController "
    //     "component.",
    //     gxf::Registrar::NoDefaultParameter(), GXF_PARAMETER_FLAGS_OPTIONAL);
    // result &= registrar->parameter(
    //     latency_pad_probe_, "latency-probe-connector", "Latency Probe Connector",
    //     "Handle to a nvidia::deepstream::NvDsProbeConnector component. The IO "
    //     "the probe will be installed on is used for latency measurement.",
    //     gxf::Registrar::NoDefaultParameter(), GXF_PARAMETER_FLAGS_OPTIONAL);
    // result &= registrar->parameter(handle_mouse_events_, "handle-mouse-events",
    //                                 "Handle Mouse Events",
    //                                 "Handle mouse events on the video renderer. "
    //                                 "Requires the 'renderer' parameter to be set.",
    //                                 true, GXF_PARAMETER_FLAGS_OPTIONAL);
    // result &= registrar->parameter(
    //     kb_input_, "kb-input", "Keyboard input",
    //     "Handle to a NvDsKeyboardInput component to read keyboard input",
    //     gxf::Registrar::NoDefaultParameter(), GXF_PARAMETER_FLAGS_OPTIONAL);
    // return nvidia::gxf::ToResultCode(result);
    // }
    // bool NvDsTilerEventHandler::handle_buffer(GstPad *pad,
    //                                         gxf::Entity buffer_data) {
    // if (!buffer_data.get<NvDsBatchMetaHandle>(BUF_DATA_KEY_VIDEO_BATCH_META)) {
    //     return true;
    // }
    // if (!buffer_data.get<GstBufferHandle>(BUF_DATA_KEY_GST_BUFFER)) {
    //     return true;
    // }
    // NvDsBatchMeta *batch_meta =
    //     *(buffer_data.get<NvDsBatchMetaHandle>(BUF_DATA_KEY_VIDEO_BATCH_META)
    //             .value());
    // OpaqueBuffer *gstbuf =
    //     *(buffer_data.get<GstBufferHandle>(BUF_DATA_KEY_GST_BUFFER).value());
    // if (active_source_index == -1 || !batch_meta) return true;
    // size_t src_cnt = 0;
    // for (NvDsUserMetaList *l = batch_meta->batch_user_meta_list; l != NULL;
    //     l = l->next) {
    //     NvDsUserMeta *user_latency_meta = reinterpret_cast<NvDsUserMeta *>(l->data);
    //     if (user_latency_meta->base_meta.meta_type ==
    //         NVDS_LATENCY_MEASUREMENT_META) {
    //     NvDsMetaCompLatency *latency_metadata =
    //         reinterpret_cast<NvDsMetaCompLatency *>(
    //             user_latency_meta->user_meta_data);
    //     if (g_str_has_prefix(latency_metadata->component_name, "nvstreammux-")) {
    //         src_cnt++;
    //     }
    //     }
    // }
    // if (src_cnt == 0) {
    //     return true;
    // }
    // latency_info.resize(src_cnt);
    // guint num_sources_in_batch =
    //     nvds_measure_buffer_latency(gstbuf, latency_info.data());
    // for (guint i = 0; i < num_sources_in_batch; i++) {
    //     if (static_cast<int>(latency_info[i].source_id) == active_source_index) {
    //     mutex.lock();
    //     active_source_latency = latency_info[i];
    //     mutex.unlock();
    //     }
    // }
    // return true;
    // }
    // gxf_result_t NvDsTilerEventHandler::initialize() {
    // if (latency_pad_probe_.try_get()) {
    //     latency_pad_probe_.try_get().value()->set_handler(this);
    //     latency_pad_probe_.try_get().value()->set_flags(NvDsProbeFlags::BUFFER);
    // }
    // return GXF_SUCCESS;
    // }

    // void NvDsTilerEventHandler::bus_callback(GstBus *bus, GstMessage *message,
    //                                         NvDsTilerEventHandler *self) {
    // GstElement *pipeline = GST_ELEMENT_PARENT(self->tiler);
    // if (GST_MESSAGE_SRC(message) == GST_OBJECT(pipeline) &&
    //     GST_MESSAGE_TYPE(message) == GST_MESSAGE_STATE_CHANGED) {
    //     GstState oldstate, newstate;
    //     gst_message_parse_state_changed(message, &oldstate, &newstate, nullptr);
    //     if (oldstate == GST_STATE_READY && newstate == GST_STATE_PAUSED) {
    //     gst_bus_disable_sync_message_emission(bus);
    //     g_signal_handler_disconnect(bus, self->sync_msg_signal_id);
    //     if (!self->create_x_window()) {
    //         return;
    //     }
    //     self->x_event_thread =
    //         std::thread([=] { self->x_event_handler_thread_func(); });
    //     std::cout << "NOTE: To expand a source in the 2D tiled display and view "
    //                 "object details, left-click on the source.\n"
    //                 "      To go back to the tiled display, right-click "
    //                 "anywhere on the window."
    //                 << std::endl;
    //     }
    // }
    // }

    bool NvDsTilerEventHandler::start() {
        started = true;
        if (!handle_mouse_events_) return true;

    // if (kb_input_.try_get() && !handle_mouse_events_.try_get().value()) {
    //     return GXF_SUCCESS;
    // }
    // if (!tiler_sink_pad_probe_.try_get()) {
    //     GXF_LOG_INFO("%s: 'tiler-out' handle not set. Not handling events", name());
    //     return GXF_SUCCESS;
    // }
    // INvDsIO *io = tiler_sink_pad_probe_.try_get().value()->get_io();
    // if (!io) {
    //     GXF_LOG_ERROR("%s: 'tiler-out' handle incorrectly set", name());
    //     return GXF_FAILURE;
    // }
    // GstPadSPtr gstpad = io->get_pad(nullptr);
    // if (!gstpad) {
    //     GXF_LOG_ERROR("%s: 'tiler-out' handle incorrectly set", name());
    //     return GXF_FAILURE;
    // }
    // tiler = reinterpret_cast<GstElement *>(
    //     gst_object_ref(GST_PAD_PARENT(gstpad.get())));
    // GstElementFactory *factory = GST_ELEMENT_GET_CLASS(tiler)->elementfactory;
    // if (g_strcmp0(GST_OBJECT_NAME(factory), "nvtilerbin")) {
    //     GXF_LOG_ERROR(
    //         "%s: 'tiler-out' handle set to incorrect IO. It should be set to the "
    //         "output of nvidia::deepstream::NvDsTiler component.",
    //         name());
    //     return GXF_FAILURE;
    // }
    // gstpad = GstPadSPtr(gst_element_get_static_pad(tiler, "src"));
    // gst_pad_add_probe(gstpad, GST_PAD_PROBE_TYPE_BUFFER, overlay_text_probe, this,
    //                     nullptr);
    // if (kb_input_.try_get()) {
    //     std::cout << "Runtime keyboard controls for tiler:\n"
    //             << "z<row-idx><col-idx> : Expand source at row-idx,col-idx in "
    //                 "the tile.\n"
    //             << "z : Go back to the tiled view when in single source mode."
    //             << std::endl;
    //     kb_event_thread =
    //         std::thread([=] { this->kb_event_handler_thread_func(); });
    // }

        if (osd) {
            // osd_property_controller = osd_property_controller_.try_get().value();
            g_print("Text disabled. Use keyboard/mouse commands to toggle source "
                        "expand and toggle text display.\n");
            osd -> set("display-text", false);
        } else {
            g_printerr(
                "%s: 'NvDsOSD not found. Drawing a lot of text in "
                "multi-stream cases might affect perf.",
                "NvDsTilerEventHandle\nr");
        }
        if (handle_mouse_events_) {
            if (!renderer) {
            g_printerr(
                "%s: 'renderer-prop-controller' handle not set. Not handling mouse "
                "events",
                "NvDsTilerEventHandler\n");
            } else {
            Display *display = XOpenDisplay(nullptr);
            if (!display) {
            g_printerr("Could not open X Display\n");
                return false;
            }
            this->display = display;
            // GstBus *bus =
            //     gst_pipeline_get_bus(GST_PIPELINE(GST_OBJECT_PARENT(tiler)));
            // gst_bus_enable_sync_message_emission(bus);
            // sync_msg_signal_id = g_signal_connect(G_OBJECT(bus), "sync-message",
            //                                         G_CALLBACK(bus_callback), this);
            }
            create_x_window();
            x_event_thread =
                std::thread([=] { x_event_handler_thread_func(); });
            g_print("NOTE: To expand a source in the 2D tiled display and view "
                        "object details, left-click on the source.\n"
                        "      To go back to the tiled display, right-click "
                        "anywhere on the window.\n");
        }
        return true;
    }


    bool NvDsTilerEventHandler::stop() {
        destroy_x_window();
        // if (tiler) gst_object_unref(tiler);
        return true;
    }

    void NvDsTilerEventHandler::set_active_source(int sourceid) {
        std::lock_guard<std::mutex> lock(mutex);
        active_source_index = sourceid;
        // if (sourceid != -1) {
        //     GstQuery *query = gst_nvquery_uri_from_streamid_new(active_source_index);
        //     GstPadSPtr sinkpad{gst_element_get_static_pad(tiler, "sink")};
        //     if (gst_pad_peer_query(sinkpad, query)) {
        //     const gchar *uri;
        //     gst_nvquery_uri_from_streamid_parse(query, &uri);
        //     active_source_uri = uri;
        //     }
        // } else {
        //     active_source_uri = "";
        // }
        tiler->set ("show-source", sourceid);
        // g_object_set(G_OBJECT(tiler), "show-source", sourceid, NULL);
        if (osd) {
            osd->set("display-text", sourceid != -1);
            // set_display_text(sourceid != -1);
        }
    }

    // GstPadProbeReturn NvDsTilerEventHandler::overlay_text_probe(
    //     GstPad *pad, GstPadProbeInfo *info, gpointer user_data) {
    // NvDsTilerEventHandler *self =
    //     reinterpret_cast<NvDsTilerEventHandler *>(user_data);
    // NvDsBatchMeta *batch_meta =
    //     gst_buffer_get_nvds_batch_meta(GST_BUFFER(info->data));
    // if (self->active_source_index == -1 || !batch_meta) return GST_PAD_PROBE_OK;
    // NvDsDisplayMeta *display_meta =
    //     nvds_acquire_display_meta_from_pool(batch_meta);
    // display_meta->num_labels = 1;
    // if (self->active_source_uri.empty()) {
    //     display_meta->text_params[0].display_text =
    //         g_strdup_printf("Source %d", self->active_source_index);
    // } else {
    //     display_meta->text_params[0].display_text =
    //         g_strdup_printf("Source: %s", self->active_source_uri.c_str());
    // }
    // display_meta->text_params[0].y_offset = 20;
    // display_meta->text_params[0].x_offset = 20;
    // display_meta->text_params[0].font_params.font_color =
    //     ((NvOSD_ColorParams){0, 1, 0, 1});
    // display_meta->text_params[0].font_params.font_size = 20;
    // display_meta->text_params[0].font_params.font_name =
    //     const_cast<char *>(FONT_NAME);
    // display_meta->text_params[0].set_bg_clr = 1;
    // display_meta->text_params[0].text_bg_clr = NvOSD_ColorParams{0, 0, 0, 1.0};
    // if (nvds_enable_latency_measurement) {
    //     display_meta->num_labels++;
    //     self->mutex.lock();
    //     display_meta->text_params[1].display_text =
    //         g_strdup_printf("Latency: %lf ms", self->active_source_latency.latency);
    //     self->mutex.unlock();
    //     display_meta->text_params[1].y_offset =
    //         (display_meta->text_params[0].y_offset * 2) +
    //         display_meta->text_params[0].font_params.font_size;
    //     display_meta->text_params[1].x_offset = 20;
    //     display_meta->text_params[1].font_params.font_color =
    //         NvOSD_ColorParams{0, 1, 0, 1};
    //     display_meta->text_params[1].font_params.font_size = 20;
    //     display_meta->text_params[1].font_params.font_name =
    //         const_cast<char *>(FONT_NAME);
    //     display_meta->text_params[1].set_bg_clr = 1;
    //     display_meta->text_params[1].text_bg_clr = NvOSD_ColorParams{0, 0, 0, 1.0};
    // }


    // nvds_add_display_meta_to_frame(
    //     nvds_get_nth_frame_meta(batch_meta->frame_meta_list, 0), display_meta);
    // return GST_PAD_PROBE_OK;
    // }

    bool NvDsTilerEventHandler::create_x_window() {
        XTextProperty xproperty;
        gchar *title;
        guint width, height;
        XSizeHints hints = {0};
        Display *display = reinterpret_cast<Display *>(this->display);
        tiler -> getProperty ("width", width);
        tiler -> getProperty ("height", height);
        // g_object_get(G_OBJECT(tiler), "width", &width, "height", &height, NULL);
        width = (width) ? width : DEFAULT_X_WINDOW_WIDTH;
        height = (height) ? height : DEFAULT_X_WINDOW_HEIGHT;
        hints.flags = PPosition | PSize;
        hints.x = 0;
        hints.y = 0;
        hints.width = width;
        hints.height = height;
        window = XCreateSimpleWindow(
            display, RootWindow(display, DefaultScreen(display)), hints.x, hints.y,
            width, height, 2, 0x00000000, 0x00000000);
        XSetNormalHints(display, window, &hints);
        title = g_strdup(APP_TITLE);
        if (XStringListToTextProperty(&title, 1, &xproperty) != 0) {
            XSetWMName(display, window, &xproperty);
            XFree(xproperty.value);
        }
        XSetWindowAttributes attr = {0};
        attr.event_mask = ButtonPress;
        XChangeWindowAttributes(display, window, CWEventMask, &attr);
        Atom wmDeleteMessage = XInternAtom(display, "WM_DELETE_WINDOW", False);
        if (wmDeleteMessage != None) {
            XSetWMProtocols(display, window, &wmDeleteMessage, 1);
        }
        XMapRaised(display, window);
        XSync(display, 1);  // discard the events for now
        set_x_window(window, renderer -> getGObject());
        return true;
    }


    static int get_source_id_from_coordinates(float x_rel, float y_rel, Element *tiler) {
        guint tile_num_rows;
        guint tile_num_columns;
        tiler -> getProperty ("rows", tile_num_rows);
        tiler -> getProperty ("columns", tile_num_columns);
        // g_object_get(G_OBJECT(tiler), "rows", &tile_num_rows, "columns",
        //             &tile_num_columns, nullptr);
        int source_id = x_rel * tile_num_columns;
        source_id += (static_cast<int>(y_rel * tile_num_rows)) * tile_num_columns;
        return source_id;
    }

    void NvDsTilerEventHandler::x_event_handler_thread_func() {
        Display *display = reinterpret_cast<Display *>(this->display);
        while (!x_event_thread_stop) {
            XEvent e = {0};
            while (XPending(display)) {
            XNextEvent(display, &e);
            switch (e.type) {
                case ButtonPress: {
                XWindowAttributes win_attr = {0};
                XButtonEvent ev = e.xbutton;
                gint source_id;
                XGetWindowAttributes(display, ev.window, &win_attr);
                tiler -> getProperty ("show-source", source_id);
                // g_object_get(G_OBJECT(tiler), "show-source", &source_id, NULL);
                if (ev.button == Button1 && source_id == -1) {
                    source_id = get_source_id_from_coordinates(
                        ev.x * 1.0 / win_attr.width, ev.y * 1.0 / win_attr.height,
                        tiler);
                    if (source_id > -1) {
                    set_active_source(source_id);
                    }
                } else if (ev.button == Button3) {
                    set_active_source(-1);
                }
                } break;
                case ClientMessage: {
                Atom wm_delete;
                wm_delete = XInternAtom(display, "WM_DELETE_WINDOW", 1);
                if (wm_delete != None && wm_delete == (Atom)e.xclient.data.l[0]) {
                    GST_ELEMENT_ERROR(tiler -> getGObject(), STREAM, FAILED,
                                    ("Output window was closed"), (NULL));
                }
                } break;
            }
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }

    }

    void NvDsTilerEventHandler::destroy_x_window() {
        Display *display = reinterpret_cast<Display *>(this->display);
        x_event_thread_stop = true;
        if (x_event_thread.joinable()) x_event_thread.join();
        x_event_thread_stop = false;
        kb_event_thread_stop = true;
        if (kb_event_thread.joinable()) kb_event_thread.join();
        kb_event_thread_stop = false;
        if (window) XDestroyWindow(display, window);
        if (display) XCloseDisplay(display);
    }

}