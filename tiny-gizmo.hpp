// This is free and unencumbered software released into the public domain.
// For more information, please refer to <http://unlicense.org>

#ifndef tinygizmo_hpp
#define tinygizmo_hpp

#include <stdint.h> // for uint32_t

namespace tinygizmo
{
    // Basic numeric types

    struct v2f { float x, y; };
    struct v3f { float x, y, z; };
    struct v4f { float x, y, z, w; };
    typedef v4f quatf;
    struct uint3 { uint32_t x, y, z; };
    struct m44f { v4f x, y, z, w; };

    // Utility object to be manipulated by the gizmos

    struct rigid_transform
    {
        rigid_transform() {}
        rigid_transform(const v4f& orientation, const v3f& position, const v3f& scale) : orientation(orientation), position(position), scale(scale) {}
        rigid_transform(const v4f& orientation, const v3f& position, float scale) : orientation(orientation), position(position), scale{ scale, scale, scale } {}
        rigid_transform(const v4f& orientation, const v3f& position) : orientation(orientation), position(position) {}

        v3f      position{ 0,0,0 };
        v4f      orientation{ 0,0,0,1 };
        v3f      scale{ 1,1,1 };

        bool     uniform_scale() const { return scale.x == scale.y && scale.x == scale.z; }
        m44f     matrix() const;
        v3f      transform_vector(const v3f& vec) const;
        v3f      transform_point(const v3f& p) const;
        v3f      detransform_point(const v3f& p) const;
        v3f      detransform_vector(const v3f& vec) const;
    };

    // input data for the gizmo's calculations

    struct camera_parameters
    {
        float yfov, near_clip, far_clip;
        v3f position;
        v4f orientation;
    };

    struct gizmo_application_state
    {
        bool mouse_left{ false };           // set to indicate that LMB is pressed
        bool hotkey_ctrl{ false };          // set to indicate that the modes for the following hotkeys are to be activated
        bool hotkey_translate{ false };     // set to indicate the hotkey to activate translation is pressed
        bool hotkey_rotate{ false };
        bool hotkey_scale{ false };
        bool hotkey_local{ false };

        float screenspace_scale{ 0.f };     // If > 0.f, the gizmos are drawn scale-invariant with a screenspace value defined here
        float snap_translation{ 0.f };      // World-scale units used for snapping translation
        float snap_scale{ 0.f };            // World-scale units used for snapping scale
        float snap_rotation{ 0.f };         // Radians used for snapping rotation quaternions (i.e. PI/8 or PI/16)

        v2f viewport_size;                  // 3d viewport used to render the view
        v3f ray_origin;                     // world-space ray origin (i.e. the camera position)
        v3f ray_direction ;                 // world-space ray direction

        camera_parameters cam;              // Used for constructing inverse view projection for raycasting onto gizmo geometry
    };

    enum class transform_mode { translate, rotate, scale };

    class gizmo_context
    {
        struct gizmo_context_impl;
        gizmo_context_impl* impl;

    public:
        gizmo_context();
        ~gizmo_context();

        transform_mode get_mode() const;  // Return the active mode being used by `transform_gizmo(...)`

        void begin(const gizmo_application_state & state); // Clear geometry buffer and update internal `gizmo_application_state` data
        void end(const gizmo_application_state & state);

        // The following functions are to be called between begin and end.

        // Fills index_buffer with faces to draw all gizmos, up to capacity. Returns desired capacity.
        // If desired capacity is greater than buffer_capacity, a larger index_buffer should be provided, and faces(...) called again.
        // Providing a null pointer for index_buffer, or a zero buffer_capacity is a quick way to discover necessary buffer size.
        size_t triangles(uint32_t* index_buffer, size_t buffer_capacity);
        size_t vertices(float* vertex_buffer, size_t stride, size_t normal_offset, size_t color_offset, size_t vertex_capacity);

        // returns true if the gizmo is hovered, or being manipulated.
        // call once for every active transform gizmo.
        // Any gizmo not named between begin and end will disappear. The manipulation state will be remembered for the next time the gizmo is activated.
        bool transform_gizmo(char const* const name, rigid_transform& t);
    };


} // end namespace tinygizmo;

#endif // end tinygizmo_hpp
