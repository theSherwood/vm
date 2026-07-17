; ModuleID = 'fs_probe.bc'
source_filename = "crates/svm-llvm/tests/fixtures/fs_probe.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-linux-gnu"

@.str = private unnamed_addr constant [3 x i8] c"fs\00", align 1
@fs = internal unnamed_addr global i32 0, align 4
@.str.1 = private unnamed_addr constant [10 x i8] c"hello.txt\00", align 1
@.str.2 = private unnamed_addr constant [11 x i8] c"hello, fs!\00", align 1
@.str.3 = private unnamed_addr constant [2 x i8] c"x\00", align 1
@.str.4 = private unnamed_addr constant [10 x i8] c"world.txt\00", align 1
@.str.5 = private unnamed_addr constant [3 x i8] c"++\00", align 1
@.str.6 = private unnamed_addr constant [10 x i8] c"trunc.txt\00", align 1
@.str.7 = private unnamed_addr constant [11 x i8] c"0123456789\00", align 1
@.str.8 = private unnamed_addr constant [10 x i8] c"../escape\00", align 1
@.str.9 = private unnamed_addr constant [10 x i8] c"/etc/pass\00", align 1
@.str.10 = private unnamed_addr constant [11 x i8] c"a/../b.txt\00", align 1
@.str.11 = private unnamed_addr constant [9 x i8] c"seed.txt\00", align 1
@.str.12 = private unnamed_addr constant [8 x i8] c"out.txt\00", align 1
@.str.13 = private unnamed_addr constant [6 x i8] c"GUEST\00", align 1
@str = private unnamed_addr constant [12 x i8] c"fs probe ok\00", align 1

; Function Attrs: nounwind uwtable
define dso_local noundef i32 @main() local_unnamed_addr #0 {
  %1 = alloca [16 x i8], align 16
  %2 = tail call i32 @__vm_cap_resolve(ptr noundef nonnull @.str, i64 noundef 2) #4
  store i32 %2, ptr @fs, align 4, !tbaa !5
  %3 = icmp slt i32 %2, 0
  br i1 %3, label %211, label %4

4:                                                ; preds = %0
  %5 = tail call i64 @__vm_host_call(i32 noundef %2, i32 noundef 0, i64 noundef ptrtoint (ptr @.str.1 to i64), i64 noundef 9, i64 noundef 26, i64 noundef 0) #4
  %6 = icmp slt i64 %5, 0
  br i1 %6, label %211, label %7

7:                                                ; preds = %4
  %8 = load i32, ptr @fs, align 4, !tbaa !5
  %9 = tail call i64 @__vm_host_call(i32 noundef %8, i32 noundef 2, i64 noundef %5, i64 noundef ptrtoint (ptr @.str.2 to i64), i64 noundef 10, i64 noundef 0) #4
  %10 = icmp eq i64 %9, 10
  br i1 %10, label %11, label %211

11:                                               ; preds = %7
  %12 = load i32, ptr @fs, align 4, !tbaa !5
  %13 = tail call i64 @__vm_host_call(i32 noundef %12, i32 noundef 4, i64 noundef %5, i64 noundef 0, i64 noundef 0, i64 noundef 0) #4
  %14 = icmp eq i64 %13, 0
  br i1 %14, label %15, label %211

15:                                               ; preds = %11
  %16 = load i32, ptr @fs, align 4, !tbaa !5
  %17 = tail call i64 @__vm_host_call(i32 noundef %16, i32 noundef 0, i64 noundef ptrtoint (ptr @.str.1 to i64), i64 noundef 9, i64 noundef 1, i64 noundef 0) #4
  %18 = icmp slt i64 %17, 0
  br i1 %18, label %211, label %19

19:                                               ; preds = %15
  %20 = load i32, ptr @fs, align 4, !tbaa !5
  %21 = tail call i64 @__vm_host_call(i32 noundef %20, i32 noundef 3, i64 noundef %17, i64 noundef 2, i64 noundef 0, i64 noundef 0) #4
  %22 = icmp eq i64 %21, 10
  br i1 %22, label %23, label %211

23:                                               ; preds = %19
  %24 = load i32, ptr @fs, align 4, !tbaa !5
  %25 = tail call i64 @__vm_host_call(i32 noundef %24, i32 noundef 3, i64 noundef %17, i64 noundef 0, i64 noundef 7, i64 noundef 0) #4
  %26 = icmp eq i64 %25, 7
  br i1 %26, label %27, label %211

27:                                               ; preds = %23
  call void @llvm.lifetime.start.p0(i64 16, ptr nonnull %1) #4
  %28 = ptrtoint ptr %1 to i64
  %29 = load i32, ptr @fs, align 4, !tbaa !5
  %30 = call i64 @__vm_host_call(i32 noundef %29, i32 noundef 1, i64 noundef %17, i64 noundef %28, i64 noundef 16, i64 noundef 0) #4
  %31 = icmp eq i64 %30, 3
  br i1 %31, label %32, label %209

32:                                               ; preds = %27
  %33 = load i8, ptr %1, align 16, !tbaa !9
  %34 = icmp ne i8 %33, 102
  %35 = getelementptr inbounds [16 x i8], ptr %1, i64 0, i64 1
  %36 = load i8, ptr %35, align 1
  %37 = icmp ne i8 %36, 115
  %38 = select i1 %34, i1 true, i1 %37
  %39 = getelementptr inbounds [16 x i8], ptr %1, i64 0, i64 2
  %40 = load i8, ptr %39, align 2
  %41 = icmp ne i8 %40, 33
  %42 = select i1 %38, i1 true, i1 %41
  br i1 %42, label %209, label %43

43:                                               ; preds = %32
  %44 = load i32, ptr @fs, align 4, !tbaa !5
  %45 = call i64 @__vm_host_call(i32 noundef %44, i32 noundef 1, i64 noundef %17, i64 noundef %28, i64 noundef 16, i64 noundef 0) #4
  %46 = icmp eq i64 %45, 0
  br i1 %46, label %47, label %209

47:                                               ; preds = %43
  %48 = load i32, ptr @fs, align 4, !tbaa !5
  %49 = call i64 @__vm_host_call(i32 noundef %48, i32 noundef 2, i64 noundef %17, i64 noundef ptrtoint (ptr @.str.3 to i64), i64 noundef 1, i64 noundef 0) #4
  %50 = icmp sgt i64 %49, -1
  br i1 %50, label %209, label %51

51:                                               ; preds = %47
  %52 = load i32, ptr @fs, align 4, !tbaa !5
  %53 = call i64 @__vm_host_call(i32 noundef %52, i32 noundef 4, i64 noundef %17, i64 noundef 0, i64 noundef 0, i64 noundef 0) #4
  %54 = icmp eq i64 %53, 0
  br i1 %54, label %55, label %209

55:                                               ; preds = %51
  %56 = load i32, ptr @fs, align 4, !tbaa !5
  %57 = call i64 @__vm_host_call(i32 noundef %56, i32 noundef 6, i64 noundef ptrtoint (ptr @.str.1 to i64), i64 noundef 9, i64 noundef ptrtoint (ptr @.str.4 to i64), i64 noundef 9) #4
  %58 = icmp eq i64 %57, 0
  br i1 %58, label %59, label %209

59:                                               ; preds = %55
  %60 = load i32, ptr @fs, align 4, !tbaa !5
  %61 = call i64 @__vm_host_call(i32 noundef %60, i32 noundef 0, i64 noundef ptrtoint (ptr @.str.1 to i64), i64 noundef 9, i64 noundef 1, i64 noundef 0) #4
  %62 = icmp sgt i64 %61, -1
  br i1 %62, label %209, label %63

63:                                               ; preds = %59
  %64 = load i32, ptr @fs, align 4, !tbaa !5
  %65 = call i64 @__vm_host_call(i32 noundef %64, i32 noundef 0, i64 noundef ptrtoint (ptr @.str.4 to i64), i64 noundef 9, i64 noundef 1, i64 noundef 0) #4
  %66 = icmp slt i64 %65, 0
  br i1 %66, label %209, label %67

67:                                               ; preds = %63
  %68 = load i32, ptr @fs, align 4, !tbaa !5
  %69 = call i64 @__vm_host_call(i32 noundef %68, i32 noundef 4, i64 noundef %65, i64 noundef 0, i64 noundef 0, i64 noundef 0) #4
  %70 = icmp eq i64 %69, 0
  br i1 %70, label %71, label %209

71:                                               ; preds = %67
  %72 = load i32, ptr @fs, align 4, !tbaa !5
  %73 = call i64 @__vm_host_call(i32 noundef %72, i32 noundef 0, i64 noundef ptrtoint (ptr @.str.4 to i64), i64 noundef 9, i64 noundef 20, i64 noundef 0) #4
  %74 = icmp slt i64 %73, 0
  br i1 %74, label %209, label %75

75:                                               ; preds = %71
  %76 = load i32, ptr @fs, align 4, !tbaa !5
  %77 = call i64 @__vm_host_call(i32 noundef %76, i32 noundef 2, i64 noundef %73, i64 noundef ptrtoint (ptr @.str.5 to i64), i64 noundef 2, i64 noundef 0) #4
  %78 = icmp eq i64 %77, 2
  br i1 %78, label %79, label %209

79:                                               ; preds = %75
  %80 = load i32, ptr @fs, align 4, !tbaa !5
  %81 = call i64 @__vm_host_call(i32 noundef %80, i32 noundef 4, i64 noundef %73, i64 noundef 0, i64 noundef 0, i64 noundef 0) #4
  %82 = icmp eq i64 %81, 0
  br i1 %82, label %83, label %209

83:                                               ; preds = %79
  %84 = load i32, ptr @fs, align 4, !tbaa !5
  %85 = call i64 @__vm_host_call(i32 noundef %84, i32 noundef 0, i64 noundef ptrtoint (ptr @.str.4 to i64), i64 noundef 9, i64 noundef 1, i64 noundef 0) #4
  %86 = load i32, ptr @fs, align 4, !tbaa !5
  %87 = call i64 @__vm_host_call(i32 noundef %86, i32 noundef 3, i64 noundef %85, i64 noundef 2, i64 noundef 0, i64 noundef 0) #4
  %88 = icmp eq i64 %87, 12
  br i1 %88, label %89, label %209

89:                                               ; preds = %83
  %90 = load i32, ptr @fs, align 4, !tbaa !5
  %91 = call i64 @__vm_host_call(i32 noundef %90, i32 noundef 4, i64 noundef %85, i64 noundef 0, i64 noundef 0, i64 noundef 0) #4
  %92 = icmp eq i64 %91, 0
  br i1 %92, label %93, label %209

93:                                               ; preds = %89
  %94 = load i32, ptr @fs, align 4, !tbaa !5
  %95 = call i64 @__vm_host_call(i32 noundef %94, i32 noundef 5, i64 noundef ptrtoint (ptr @.str.4 to i64), i64 noundef 9, i64 noundef 0, i64 noundef 0) #4
  %96 = icmp eq i64 %95, 0
  br i1 %96, label %97, label %209

97:                                               ; preds = %93
  %98 = load i32, ptr @fs, align 4, !tbaa !5
  %99 = call i64 @__vm_host_call(i32 noundef %98, i32 noundef 0, i64 noundef ptrtoint (ptr @.str.4 to i64), i64 noundef 9, i64 noundef 1, i64 noundef 0) #4
  %100 = icmp sgt i64 %99, -1
  br i1 %100, label %209, label %101

101:                                              ; preds = %97
  %102 = load i32, ptr @fs, align 4, !tbaa !5
  %103 = call i64 @__vm_host_call(i32 noundef %102, i32 noundef 5, i64 noundef ptrtoint (ptr @.str.4 to i64), i64 noundef 9, i64 noundef 0, i64 noundef 0) #4
  %104 = icmp sgt i64 %103, -1
  br i1 %104, label %209, label %105

105:                                              ; preds = %101
  %106 = load i32, ptr @fs, align 4, !tbaa !5
  %107 = call i64 @__vm_host_call(i32 noundef %106, i32 noundef 0, i64 noundef ptrtoint (ptr @.str.6 to i64), i64 noundef 9, i64 noundef 27, i64 noundef 0) #4
  %108 = icmp slt i64 %107, 0
  br i1 %108, label %209, label %109

109:                                              ; preds = %105
  %110 = load i32, ptr @fs, align 4, !tbaa !5
  %111 = call i64 @__vm_host_call(i32 noundef %110, i32 noundef 2, i64 noundef %107, i64 noundef ptrtoint (ptr @.str.7 to i64), i64 noundef 10, i64 noundef 0) #4
  %112 = icmp eq i64 %111, 10
  br i1 %112, label %113, label %209

113:                                              ; preds = %109
  %114 = load i32, ptr @fs, align 4, !tbaa !5
  %115 = call i64 @__vm_host_call(i32 noundef %114, i32 noundef 7, i64 noundef %107, i64 noundef 4, i64 noundef 0, i64 noundef 0) #4
  %116 = icmp eq i64 %115, 0
  br i1 %116, label %117, label %209

117:                                              ; preds = %113
  %118 = load i32, ptr @fs, align 4, !tbaa !5
  %119 = call i64 @__vm_host_call(i32 noundef %118, i32 noundef 8, i64 noundef %107, i64 noundef 0, i64 noundef 0, i64 noundef 0) #4
  %120 = icmp eq i64 %119, 0
  br i1 %120, label %121, label %209

121:                                              ; preds = %117
  %122 = load i32, ptr @fs, align 4, !tbaa !5
  %123 = call i64 @__vm_host_call(i32 noundef %122, i32 noundef 3, i64 noundef %107, i64 noundef 2, i64 noundef 0, i64 noundef 0) #4
  %124 = icmp eq i64 %123, 4
  br i1 %124, label %125, label %209

125:                                              ; preds = %121
  %126 = load i32, ptr @fs, align 4, !tbaa !5
  %127 = call i64 @__vm_host_call(i32 noundef %126, i32 noundef 7, i64 noundef %107, i64 noundef 6, i64 noundef 0, i64 noundef 0) #4
  %128 = icmp eq i64 %127, 0
  br i1 %128, label %129, label %209

129:                                              ; preds = %125
  %130 = load i32, ptr @fs, align 4, !tbaa !5
  %131 = call i64 @__vm_host_call(i32 noundef %130, i32 noundef 3, i64 noundef %107, i64 noundef 0, i64 noundef 3, i64 noundef 0) #4
  %132 = icmp eq i64 %131, 3
  br i1 %132, label %133, label %209

133:                                              ; preds = %129
  %134 = load i32, ptr @fs, align 4, !tbaa !5
  %135 = call i64 @__vm_host_call(i32 noundef %134, i32 noundef 1, i64 noundef %107, i64 noundef %28, i64 noundef 16, i64 noundef 0) #4
  %136 = icmp eq i64 %135, 3
  br i1 %136, label %137, label %209

137:                                              ; preds = %133
  %138 = load i8, ptr %1, align 16, !tbaa !9
  %139 = icmp ne i8 %138, 51
  %140 = load i8, ptr %35, align 1
  %141 = icmp ne i8 %140, 0
  %142 = select i1 %139, i1 true, i1 %141
  %143 = load i8, ptr %39, align 2
  %144 = icmp ne i8 %143, 0
  %145 = select i1 %142, i1 true, i1 %144
  br i1 %145, label %209, label %146

146:                                              ; preds = %137
  %147 = load i32, ptr @fs, align 4, !tbaa !5
  %148 = call i64 @__vm_host_call(i32 noundef %147, i32 noundef 4, i64 noundef %107, i64 noundef 0, i64 noundef 0, i64 noundef 0) #4
  %149 = icmp eq i64 %148, 0
  br i1 %149, label %150, label %209

150:                                              ; preds = %146
  %151 = load i32, ptr @fs, align 4, !tbaa !5
  %152 = call i64 @__vm_host_call(i32 noundef %151, i32 noundef 0, i64 noundef ptrtoint (ptr @.str.6 to i64), i64 noundef 9, i64 noundef 1, i64 noundef 0) #4
  %153 = icmp slt i64 %152, 0
  br i1 %153, label %209, label %154

154:                                              ; preds = %150
  %155 = load i32, ptr @fs, align 4, !tbaa !5
  %156 = call i64 @__vm_host_call(i32 noundef %155, i32 noundef 7, i64 noundef %152, i64 noundef 0, i64 noundef 0, i64 noundef 0) #4
  %157 = icmp sgt i64 %156, -1
  br i1 %157, label %209, label %158

158:                                              ; preds = %154
  %159 = load i32, ptr @fs, align 4, !tbaa !5
  %160 = call i64 @__vm_host_call(i32 noundef %159, i32 noundef 4, i64 noundef %152, i64 noundef 0, i64 noundef 0, i64 noundef 0) #4
  %161 = icmp eq i64 %160, 0
  br i1 %161, label %162, label %209

162:                                              ; preds = %158
  %163 = load i32, ptr @fs, align 4, !tbaa !5
  %164 = call i64 @__vm_host_call(i32 noundef %163, i32 noundef 5, i64 noundef ptrtoint (ptr @.str.6 to i64), i64 noundef 9, i64 noundef 0, i64 noundef 0) #4
  %165 = icmp eq i64 %164, 0
  br i1 %165, label %166, label %209

166:                                              ; preds = %162
  %167 = load i32, ptr @fs, align 4, !tbaa !5
  %168 = call i64 @__vm_host_call(i32 noundef %167, i32 noundef 0, i64 noundef ptrtoint (ptr @.str.8 to i64), i64 noundef 9, i64 noundef 18, i64 noundef 0) #4
  %169 = icmp sgt i64 %168, -1
  br i1 %169, label %209, label %170

170:                                              ; preds = %166
  %171 = load i32, ptr @fs, align 4, !tbaa !5
  %172 = call i64 @__vm_host_call(i32 noundef %171, i32 noundef 0, i64 noundef ptrtoint (ptr @.str.9 to i64), i64 noundef 9, i64 noundef 1, i64 noundef 0) #4
  %173 = icmp sgt i64 %172, -1
  br i1 %173, label %209, label %174

174:                                              ; preds = %170
  %175 = load i32, ptr @fs, align 4, !tbaa !5
  %176 = call i64 @__vm_host_call(i32 noundef %175, i32 noundef 0, i64 noundef ptrtoint (ptr @.str.10 to i64), i64 noundef 10, i64 noundef 18, i64 noundef 0) #4
  %177 = icmp sgt i64 %176, -1
  br i1 %177, label %209, label %178

178:                                              ; preds = %174
  %179 = load i32, ptr @fs, align 4, !tbaa !5
  %180 = call i64 @__vm_host_call(i32 noundef %179, i32 noundef 0, i64 noundef ptrtoint (ptr @.str.11 to i64), i64 noundef 8, i64 noundef 1, i64 noundef 0) #4
  %181 = icmp sgt i64 %180, -1
  br i1 %181, label %182, label %207

182:                                              ; preds = %178
  %183 = load i32, ptr @fs, align 4, !tbaa !5
  %184 = call i64 @__vm_host_call(i32 noundef %183, i32 noundef 1, i64 noundef %180, i64 noundef %28, i64 noundef 16, i64 noundef 0) #4
  %185 = icmp eq i64 %184, 4
  br i1 %185, label %186, label %209

186:                                              ; preds = %182
  %187 = load <4 x i8>, ptr %1, align 16
  %188 = freeze <4 x i8> %187
  %189 = bitcast <4 x i8> %188 to i32
  %190 = icmp eq i32 %189, 1145390419
  br i1 %190, label %191, label %209

191:                                              ; preds = %186
  %192 = load i32, ptr @fs, align 4, !tbaa !5
  %193 = call i64 @__vm_host_call(i32 noundef %192, i32 noundef 4, i64 noundef %180, i64 noundef 0, i64 noundef 0, i64 noundef 0) #4
  %194 = icmp eq i64 %193, 0
  br i1 %194, label %195, label %209

195:                                              ; preds = %191
  %196 = load i32, ptr @fs, align 4, !tbaa !5
  %197 = call i64 @__vm_host_call(i32 noundef %196, i32 noundef 0, i64 noundef ptrtoint (ptr @.str.12 to i64), i64 noundef 7, i64 noundef 26, i64 noundef 0) #4
  %198 = icmp slt i64 %197, 0
  br i1 %198, label %209, label %199

199:                                              ; preds = %195
  %200 = load i32, ptr @fs, align 4, !tbaa !5
  %201 = call i64 @__vm_host_call(i32 noundef %200, i32 noundef 2, i64 noundef %197, i64 noundef ptrtoint (ptr @.str.13 to i64), i64 noundef 5, i64 noundef 0) #4
  %202 = icmp eq i64 %201, 5
  br i1 %202, label %203, label %209

203:                                              ; preds = %199
  %204 = load i32, ptr @fs, align 4, !tbaa !5
  %205 = call i64 @__vm_host_call(i32 noundef %204, i32 noundef 4, i64 noundef %197, i64 noundef 0, i64 noundef 0, i64 noundef 0) #4
  %206 = icmp eq i64 %205, 0
  br i1 %206, label %207, label %209

207:                                              ; preds = %203, %178
  %208 = call i32 @puts(ptr nonnull dereferenceable(1) @str)
  br label %209

209:                                              ; preds = %203, %199, %195, %191, %186, %182, %174, %170, %166, %162, %158, %154, %150, %146, %137, %133, %129, %125, %121, %117, %113, %109, %105, %101, %97, %93, %89, %83, %79, %75, %71, %67, %63, %59, %55, %51, %47, %43, %32, %27, %207
  %210 = phi i32 [ 0, %207 ], [ 8, %27 ], [ 9, %32 ], [ 10, %43 ], [ 11, %47 ], [ 12, %51 ], [ 13, %55 ], [ 14, %59 ], [ 15, %63 ], [ 16, %67 ], [ 17, %71 ], [ 18, %75 ], [ 19, %79 ], [ 20, %83 ], [ 21, %89 ], [ 22, %93 ], [ 23, %97 ], [ 24, %101 ], [ 34, %105 ], [ 35, %109 ], [ 36, %113 ], [ 37, %117 ], [ 38, %121 ], [ 39, %125 ], [ 40, %129 ], [ 41, %133 ], [ 42, %137 ], [ 43, %146 ], [ 44, %150 ], [ 45, %154 ], [ 46, %158 ], [ 47, %162 ], [ 25, %166 ], [ 26, %170 ], [ 27, %174 ], [ 28, %182 ], [ 29, %186 ], [ 30, %191 ], [ 31, %195 ], [ 32, %199 ], [ 33, %203 ]
  call void @llvm.lifetime.end.p0(i64 16, ptr nonnull %1) #4
  br label %211

211:                                              ; preds = %209, %4, %7, %11, %15, %19, %23, %0
  %212 = phi i32 [ 1, %0 ], [ %210, %209 ], [ 2, %4 ], [ 3, %7 ], [ 4, %11 ], [ 5, %15 ], [ 6, %19 ], [ 7, %23 ]
  ret i32 %212
}

declare i32 @__vm_cap_resolve(ptr noundef, i64 noundef) local_unnamed_addr #1

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.start.p0(i64 immarg, ptr nocapture) #2

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.end.p0(i64 immarg, ptr nocapture) #2

declare i64 @__vm_host_call(i32 noundef, i32 noundef, i64 noundef, i64 noundef, i64 noundef, i64 noundef) local_unnamed_addr #1

; Function Attrs: nofree nounwind
declare noundef i32 @puts(ptr nocapture noundef readonly) local_unnamed_addr #3

attributes #0 = { nounwind uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #1 = { "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #2 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
attributes #3 = { nofree nounwind }
attributes #4 = { nounwind }

!llvm.module.flags = !{!0, !1, !2, !3}
!llvm.ident = !{!4}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{!"Ubuntu clang version 18.1.3 (1ubuntu1)"}
!5 = !{!6, !6, i64 0}
!6 = !{!"int", !7, i64 0}
!7 = !{!"omnipotent char", !8, i64 0}
!8 = !{!"Simple C/C++ TBAA"}
!9 = !{!7, !7, i64 0}
